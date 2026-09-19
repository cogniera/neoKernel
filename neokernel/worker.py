"""Trusted bootstrap passed via -c; this file is never imported by the candidate."""

import json
import os
import sys
import time
import traceback
import struct
import tempfile


def main():
    wire = sys.stdout
    sys.stdout = sys.stderr
    binary = len(sys.argv) > 4 and sys.argv[4] == "binary"

    def send(value):
        if binary:
            if value["kind"] == "step":
                tokens = value["tokens"]
                payload = b"NKST" + struct.pack("<Idd", len(tokens), value["child_sent_s"], value["child_cpu_ms"])
                payload += struct.pack(f"<{len(tokens)}q", *tokens)
            else:
                encoded = json.dumps(value, allow_nan=False).encode("utf-8")
                payload = b"NKJS" + struct.pack("<I", len(encoded)) + encoded
            offset = 0
            while offset < len(payload):
                offset += wire.buffer.write(payload[offset:])
            wire.buffer.flush()
        else:
            wire.write(json.dumps(value, allow_nan=False) + "\n")
            wire.flush()

    try:
        import torch
        threads_before = torch.get_num_threads()
        torch.set_num_threads(8)
        quota_path = "/sys/fs/cgroup/cpu.max"
        quota = open(quota_path).read().strip() if os.path.exists(quota_path) else None
        diagnostics = {"os_cpu_count": os.cpu_count(), "cpu_max": quota,
                       "torch_threads_before": threads_before, "torch_num_threads": torch.get_num_threads()}
        diagnostics["cgroup_v1"] = {path: open(path).read().strip() for path in [
            '/sys/fs/cgroup/cpu/cpu.cfs_quota_us', '/sys/fs/cgroup/cpu/cpu.cfs_period_us',
            '/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us', '/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us']
            if os.path.exists(path)}
        print(json.dumps({"child_cpu": diagnostics}), file=sys.stderr, flush=True)
        sys.path.insert(0, sys.argv[1])
        os.chdir(sys.argv[1])
        if len(sys.argv) > 3 and sys.argv[3] == "validate_rmsnorm":
            from kernels.rmsnorm import rms_norm
            torch.manual_seed(1234)
            with torch.inference_mode():
                for rows, width in [(1, 128), (32, 128), (1, 2560), (16, 2560), (512, 2560)]:
                    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
                    reference = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * weight
                    actual = rms_norm(x, weight, 1e-6)
                    torch.testing.assert_close(actual, reference, rtol=.016, atol=.016)
            send({"kind": "validated", "kernel": "rmsnorm"})
            return
        from engine import Engine
        engine = Engine(sys.argv[2])
        lifetime_peak_mem_bytes = torch.cuda.max_memory_allocated()
        send({"kind": "ready", "diagnostics": diagnostics})
        for line in sys.stdin:
            request = json.loads(line)
            if request["kind"] == "generate":
                torch.cuda.synchronize()
                lifetime_peak_mem_bytes = max(lifetime_peak_mem_bytes, torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
                generation = iter(engine.generate(request["prompt"], request["N"]))
                while True:
                    cpu_start = time.process_time()
                    try:
                        tokens = next(generation)
                    except StopIteration:
                        break
                    cpu_ms = (time.process_time()-cpu_start)*1000
                    if type(tokens) is not list or any(type(t) is not int for t in tokens):
                        raise TypeError("candidate must yield Python lists of Python ints")
                    send({"kind": "step", "tokens": tokens, "child_cpu_ms": cpu_ms,
                          "child_sent_s": time.perf_counter()})
                torch.cuda.synchronize()
                sample_peak_mem_bytes = torch.cuda.max_memory_allocated()
                lifetime_peak_mem_bytes = max(lifetime_peak_mem_bytes, sample_peak_mem_bytes)
                send({"kind": "done", "peak_mem_bytes": sample_peak_mem_bytes})
            elif request["kind"] == "memory":
                torch.cuda.synchronize()
                lifetime_peak_mem_bytes = max(lifetime_peak_mem_bytes, torch.cuda.max_memory_allocated())
                send({"kind": "memory", "peak_mem_bytes": lifetime_peak_mem_bytes})
            elif request["kind"] == "profile":
                stream = engine.generate(request["prompt"], request["N"])
                next(stream)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                    started = time.perf_counter()
                    with torch.profiler.record_function('engine_decode_step'):
                        next(stream)
                    wall_ms = (time.perf_counter() - started) * 1000
                events = []
                byte_estimates = {}
                for event in prof.events():
                    if event.device_type == torch.autograd.DeviceType.CUDA:
                        events.append({"name": event.name, "start_us": event.time_range.start,
                                       "end_us": event.time_range.end})
                    # A single GEMM kernel with explicit 2-D inputs has a derivable
                    # BF16 payload estimate. Avoid assigning bytes from names alone.
                    if event.name == "aten::mm" and len(event.input_shapes) >= 2 and len(event.kernels) == 1:
                        a, b = event.input_shapes[:2]
                        if len(a) == len(b) == 2 and a[1] == b[0]:
                            name = event.kernels[0].name
                            moved = 2 * (a[0]*a[1] + b[0]*b[1] + a[0]*b[1])
                            byte_estimates[name] = byte_estimates.get(name, 0) + moved
                stream.close()
                with tempfile.TemporaryDirectory() as trace_dir:
                    trace_path = os.path.join(trace_dir, 'decode.json')
                    prof.export_chrome_trace(trace_path)
                    with open(trace_path) as trace_file:
                        trace = json.load(trace_file)
                traced = trace['traceEvents']
                kernel_events = [e for e in traced if e.get('cat') == 'kernel']
                events = [{'name': e['name'], 'start_us': e['ts'], 'end_us': e['ts']+e['dur']}
                          for e in kernel_events]
                region = next(e for e in traced if e.get('name') == 'engine_decode_step')
                runtime = [e['name'] for e in traced if e.get('cat') == 'cuda_runtime'
                           and region['ts'] <= e.get('ts', -1) <= region['ts']+region['dur']]
                send({"kind": "profile", "step_wall_ms": wall_ms, "events": events,
                      "byte_estimates": byte_estimates, 'kernel_count': len(kernel_events),
                      'graph_launch_count': sum('cudaGraphLaunch' in name for name in runtime),
                      'sync_calls': [name for name in runtime if 'Synchronize' in name],
                      'runtime_calls': runtime, 'trace': trace})
            else:
                raise ValueError("unknown worker request")
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        send({"kind": "error", "message": f"{type(exc).__name__}: {exc}"[:2000]})


if __name__ == "__main__":
    main()
