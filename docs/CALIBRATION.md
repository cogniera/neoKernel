# Calibration

Official reference: Dryft run 4877cddd. Local-native measurement: unchanged starter, H100, three samples per public workload, paired with the candidate in alternating order.

Source run: 2026-09-19T18:28:53.827905+00:00. Existing Git SHA: c2405f19fae577539969c5face3914b116757ae6. Engine archive SHA256: 9013fe95cf31fd574b947acad03938d3491979cebb2d0693bad50ae1bf271504.

| Workload | Native tok/s | Official tok/s | Host factor | Native TTFT ms | Official TTFT ms | TTFT factor | Native TPOT ms | Official TPOT ms | TPOT factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| public-0 | 39.07 | 57.00 | 0.6855 | 28.64 | 22.01 | 1.3014 | 25.40 | 17.41 | 1.4591 |
| public-1 | 126.77 | 150.00 | 0.8451 | 203.31 | 202.79 | 1.0026 | 26.02 | 21.01 | 1.2386 |
| public-2 | 619.84 | 671.20 | 0.9235 | 192.29 | 192.63 | 0.9982 | 24.50 | 22.51 | 1.0886 |

Every throughput ratio within 15 percent: False. All benchmark gates passed: True.

Ratios are local divided by official. The table reports residual differences without normalizing the measurements. The CLI divides later H100 public measurements by these baseline ratios to display Dryft-equivalent estimates when the 15 percent criterion passes or the operator accepts the host residual. Those estimates do not affect correctness, gates, or keep decisions and do not predict hidden-workload leaderboard scores. Latency gates use the paired local native medians, never the official values.

Parent CPU diagnostics: `{"cgroup_v1": {"/sys/fs/cgroup/cpu/cpu.cfs_period_us": "1000000", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "24000000"}, "cpu_max": null, "os_cpu_count": 24, "torch_num_threads": 8, "torch_threads_before": 8}`. Transport: binary.

| Workload | Native pipe mean ms | Native pipe max ms | Native CPU/decode mean ms | Native CPU/decode max ms | Child cgroup cpu.max |
| --- | ---: | ---: | ---: | ---: | --- |
| public-0 | 0.38970318750009003 | 0.7679610000010939 | 24.3010752688172 | 39.99999999999915 | None |
| public-1 | 0.38390191666641077 | 0.9818420000016204 | 25.161290322580605 | 30.000000000001137 | None |
| public-2 | 0.4759104244792904 | 0.8646619999979066 | 23.333333333333336 | 40.000000000000924 | None |

The judge uses random non-special token IDs rather than Dryft's private corpus. Its parent owns arrival timestamps and replay; the candidate loads and warms up once per workload, sees distinct prompts, and reports lifetime GPU peak memory after samples. The independent native model stays resident in the parent. These differences and uncontrolled GPU timing mean calibration is approximate.

Calibration stopped after the CPU-8 rerun because public-0 is 31.45 percent below official throughput and public-1 is 15.49 percent below. Steps 6â€“8 were not started. No commit, push, or engine change was made.

The preceding CPU-8 JSON run measured native throughput 43.21 / 133.61 / 631.75 tok/s. Candidate pipe latency peaked at 1.3971 and 2.0294 ms, triggering the switch to unbuffered binary frames. The binary run above used a different container and did not improve throughput; these runs do not isolate transport from host variation. Native binary pipe maxima are below 1 ms on every shape; candidate public-2 still has a 1.7417 ms maximum (0.4422 ms mean).

The cgroup v2 cpu.max file is absent. Cgroup v1 exposes cpu.cfs_quota_us=24000000 and cpu.cfs_period_us=1000000, equivalent to 24 CPU units; os.cpu_count() also reports 24. The Modal request is cpu=8.0, memory=32768, and parent and child use eight PyTorch threads. This does not show a low exposed CPU quota. Native decode process CPU time averages 24.30 / 25.16 / 23.33 ms, close to wall TPOT 25.40 / 26.02 / 24.50 ms. That is consistent with substantial host execution cost rather than pipe transfer explaining the gap, but does not establish the cause. Process CPU time observations are quantized in approximately 10 ms increments in this environment; individual-step maxima are coarse.

Validation: 47 CPU tests passed with no skips. Prior L4 correctness near-tie counts were 0 / 0 / 9. In the final three-sample H100 run, candidate counts are 0 / 0 / 10 and native counts are 0 / 0 / 12; all correctness checks and paired local benchmark gates pass.

Cumulative reported GPU time: L4 96.535 s, H100 268.558 s. Estimated allocation time including startup/idle: L4 110.829 s, H100 299.015 s. Estimated tier costs including CPU and RAM: L4 $0.03538, H100 $0.37491; CPU-only setup $0.00834; cumulative $0.41863. These are estimates, not billing records.

Calibration accepted by the operator: public-2 meets the 15 percent target; public-0/1 do not because of residual host single-core speed, supported by the CPU-time evidence above. Accepted host factors are 0.6855 / 0.8451 / 0.9235. Bench divides local throughput by the factor and labels the result an estimate. Factors refresh automatically from each new native measurement; they may change as hosts change. Local TTFT and TPOT gates are tightened to 1.05 times local native, compared with official 1.10. Proceeding with steps 6–9.

Step 6 completed: profile, gauge, and log rendered in Windows. The first profile response exceeded the token-message size limit; profile-only responses now allow 64 MiB, while token messages retain the 1 MiB bound. A regression test covers both limits. The successful L4 profile reported step_wall_ms=132.575, sum_kernel_ms=38.047, gap_ms=94.528, and estimated weight floor=2.401 ms. Profiler overhead is included; these L4 timings are not H100 throughput measurements. All 49 CPU tests pass without skips. Estimated cumulative spend before the attended iteration is $0.8330, including the conservative $0.3827 allowance for the failed profile.
