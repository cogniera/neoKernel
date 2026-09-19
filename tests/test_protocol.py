"""Real subprocess/pipe tests with a standard-library stand-in worker."""

import tempfile
import time
import unittest
import io
import json
import sys
from types import ModuleType
from pathlib import Path
from unittest.mock import patch

from neokernel.judge import CandidateError, Child, child_launch_options
from neokernel.schema import Workload

WORKER = '''import json, sys, time
print(json.dumps({"kind": "ready"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    mode = sys.argv[2]
    if mode == "hang":
        time.sleep(60)
    n = request["N"] + (1 if mode == "extra" else -1 if mode == "early" else 0)
    for i in range(n):
        tokens = [True] if mode == "bool" else [i]
        print(json.dumps({"kind": "step", "tokens": tokens, "timestamp_s": -123456}), flush=True)
    print(json.dumps({"kind": "done", "peak_mem_bytes": 8}), flush=True)
'''


class ProtocolTests(unittest.TestCase):
    def test_large_profile_response_keeps_token_message_limit(self):
        source = "import json; print(json.dumps({'kind':'ready', 'trace':'x'*(1024*1024)}), flush=True)"
        for mode in ['profile', 'engine']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                with patch('neokernel.judge.Path.read_text', return_value=source):
                    child = Child(Path(tmp), 'unused', mode=mode)
                try:
                    if mode == 'profile':
                        self.assertEqual(child.receive(time.perf_counter()+5)[1]['kind'], 'ready')
                    else:
                        with self.assertRaisesRegex(CandidateError, 'oversized wire message'):
                            child.receive(time.perf_counter()+5)
                finally:
                    child.close()

    def test_binary_child_roundtrip(self):
        binary_worker = '''import json, sys, struct, time
def send(value):
    data = json.dumps(value).encode()
    sys.stdout.buffer.write(b'NKJS' + struct.pack('<I', len(data)) + data)
    sys.stdout.buffer.flush()
send({'kind': 'ready'})
for line in sys.stdin:
    request = json.loads(line)
    for i in range(request['N']):
        sys.stdout.buffer.write(b'NKST' + struct.pack('<Iddq', 1, time.perf_counter(), .5, i))
        sys.stdout.buffer.flush()
    send({'kind': 'done', 'peak_mem_bytes': 8})
'''
        with tempfile.TemporaryDirectory() as tmp:
            with patch('neokernel.judge.Path.read_text', return_value=binary_worker):
                child = Child(Path(tmp), 'unused', transport='binary')
            try:
                self.assertEqual(child.receive(time.perf_counter()+5)[1]['kind'], 'ready')
                result = child.sample([[1]], Workload('tiny', 1, 1, 3), 8)
                self.assertEqual(result.tokens, [[0], [1], [2]])
                self.assertEqual(result.child_cpu_ms, [.5, .5, .5])
                self.assertEqual(len(result.pipe_overhead_ms), 3)
                self.assertTrue(all(v >= 0 for v in result.pipe_overhead_ms))
            finally:
                child.close()

    def test_binary_partial_frames_and_parent_timestamp(self):
        import struct
        from neokernel.judge import read_binary_message
        class Fragmented(io.BytesIO):
            def read(self, count=-1):
                return super().read(min(count, 3))
        frame = b'NKST' + struct.pack('<Idd2q', 2, -123456, 7.5, 3, 4)
        with patch('neokernel.judge.time.perf_counter', return_value=10):
            stamp, message = read_binary_message(Fragmented(frame))
        self.assertEqual(stamp, 10)
        self.assertEqual(message['tokens'], [3, 4])
        self.assertEqual(message['child_cpu_ms'], 7.5)
        with self.assertRaises(EOFError):
            read_binary_message(Fragmented(frame[:-1]))

    def test_launch_options_for_windows_and_linux(self):
        self.assertEqual(child_launch_options("nt"), {"start_new_session": False, "creationflags": 0x08000000})
        self.assertEqual(child_launch_options("posix"), {"start_new_session": True})

    def test_worker_reports_lifetime_peak_after_samples(self):
        from neokernel import worker
        fake_torch = ModuleType("torch")
        from unittest.mock import Mock
        fake_torch.cuda = Mock()
        fake_torch.get_num_threads = Mock(return_value=8)
        fake_torch.set_num_threads = Mock()
        fake_torch.cuda.max_memory_allocated.side_effect = [100, 100, 50, 50, 60, 60]
        fake_engine = ModuleType("engine")
        generated = []
        class Engine:
            def __init__(self, model_path):
                pass
            def generate(self, prompt, N):
                for i in range(N):
                    generated.append(i)
                    yield [i]
        fake_engine.Engine = Engine
        requests = [{"kind": "generate", "prompt": [[1]], "N": 2},
                    {"kind": "generate", "prompt": [[2]], "N": 2}, {"kind": "memory"}]
        output = io.StringIO()
        with patch.dict(sys.modules, {"torch": fake_torch, "engine": fake_engine}), \
                patch.object(sys, "argv", ["worker", ".", "unused"]), \
                patch.object(sys, "path", sys.path.copy()), patch("neokernel.worker.os.chdir"), \
                patch.object(sys, "stdin", io.StringIO("\n".join(map(json.dumps, requests)))), \
                patch.object(sys, "stdout", output):
            worker.main()
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(messages[-1], {"kind": "memory", "peak_mem_bytes": 100})
        self.assertEqual([m["peak_mem_bytes"] for m in messages if m["kind"] == "done"], [50, 60])
        self.assertEqual(len(generated), 4)
        self.assertTrue(all({"kind", "tokens", "child_sent_s", "child_cpu_ms"} == set(m) for m in messages if m["kind"] == "step"))

    def child(self, root, mode):
        with patch("neokernel.judge.Path.read_text", return_value=WORKER):
            child = Child(root, mode)
        _, ready = child.receive(time.perf_counter()+5)
        self.assertEqual(ready["kind"], "ready")
        return child

    def test_stream_timing_and_fresh_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = self.child(Path(tmp), "normal")
            try:
                for prompt in [[[1, 2]], [[2, 3]]]:
                    result = child.sample(prompt, Workload("tiny", 1, 2, 3), 8)
                    self.assertEqual(result.tokens, [[0], [1], [2]])
                    self.assertGreater(result.ttft_s, 0)
                    self.assertEqual(result.total_s, result.step_times_s[-1])
                    self.assertAlmostEqual(result.tpot_s, (result.total_s-result.ttft_s)/2)
            finally:
                child.close()
            self.assertIsNotNone(child.process.returncode)

    def test_stream_shape_failures(self):
        for mode in ["extra", "early", "bool"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                child = self.child(Path(tmp), mode)
                try:
                    with self.assertRaises(CandidateError):
                        child.sample([[1]], Workload("tiny", 1, 1, 3), 8)
                finally:
                    child.close()

    def test_deadline_terminates_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = self.child(Path(tmp), "hang")
            try:
                with self.assertRaises(TimeoutError):
                    child.sample([[1]], Workload("tiny", 1, 1, 3), 8, deadline=time.perf_counter()+.05)
            finally:
                child.close()
            self.assertIsNotNone(child.process.returncode)


if __name__ == "__main__":
    unittest.main()
