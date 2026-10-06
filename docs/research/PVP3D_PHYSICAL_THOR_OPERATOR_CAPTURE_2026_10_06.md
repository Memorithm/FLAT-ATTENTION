# PVP3d physical Thor operator capture — 2026-10-06

Status: retained device-specific observation; no runtime-selection promotion.
The external admission method remains subject to review and is not a reusable
RemoteOps admission capability.

## Actual execution and identity

The operator-controlled reservation ran on Thor from 21:34:50 to 21:35:34 UTC
(23:34:50–23:35:34 Europe/Paris). Source was a clean Git checkout of
`50c0230cc645bd44f1520e2c7194878a4b13009c`, compiled with Rust 1.89.0 / LLVM
20.1.7 on aarch64. The benchmark binary SHA-256 is
`f4b7f2a4e3d31147c80c2cb724a981c48f6c402002dbb4bd036f2d6a93083d81`.
Actual adapter: NVIDIA Tegra NVIDIA Thor, Vulkan, driver 580.00.
Only the installed vendor driver was used; no NVML or CUDA execution API.

The preregistered PVP3d grid, candidate implementations, five warmups, thirty
repetitions, balanced six-permutation order and timing scope were unchanged.
Full candidate readback passed the CPU oracle before and after each panel.
The separate explicit-ANF suite passed all 24 banks across four kernels:
96 exact candidate comparisons plus coefficient recovery on this physical GPU.

The ANF test was built with `cargo +1.89.0 test --locked --release --features
wgpu --test pvp_anf_bank --no-run` in this freshly cloned isolated checkout.
Its target directory contained exactly one matching executable, whose filename
agrees with the retained compiler output. The supplemental executable identity
records its SHA-256 observed after capture, before any further build. This
establishes the actual single-build context; the historical controller's
`find | head` is not a safe selection rule for a reused target directory.

## Reservation, monitoring and restoration

The operator held `/dev/nvidia0` with `flock` and verified independent contention.
The active CLM, CLM-encoder and Viggle services were stopped. Temporary runtime
systemd drop-ins refused manual starts and disabled restart during the window.
An independent 30-minute restoration timer was installed before suspension;
normal cleanup removed the drop-ins, started the original three services and
cancelled the timer. Captured final states are active/running with original
restart policy and no manual-start refusal. The qualification unit exited zero.

During execution, the Rust RemoteOps device-user probe scanned `/dev/nvidia0`
and `/dev/dri/renderD128`. The controller required observed scans and attributed
every visible device user to its qualification cgroup. Polling occurred at
approximately one-second intervals. Partial observations were retried only
outside measured execution; a partial or foreign user during measurement would
reject the attempt. The raw record includes two partial observations outside
the measurement phase rather than concealing them.

This is a cooperative lock with controlled services and sampled observation,
not a hostile-process fence, hardware utilization meter, cryptographic
attestation or proof that unsampled transient activity is impossible. Other
host CPUs were not exclusively reserved; load averages are retained.
The source of the one-off lifecycle controller and JSON checker is retained
as text for audit; it is not new model/runtime infrastructure.

A first attempt passed ANF but encountered an incomplete post-ANF scan and
restored services before timing began. Its records remain under `attempt-1/`.
The second attempt retained all 63 frozen candidate rows: 45 measured rows
(15 complete geometries) and 18 rows rejected by the common dispatch limit
(six complete geometries). All 1,350 raw timing samples are retained.

## Observed resident wall medians

Time includes encode, submit and completed wait, with reset, upload, pipeline
creation and readback outside the timer. Ratios use vec4 median divided by the
candidate median. Values above one are lower observed candidate wall latency;
these are single-run descriptive comparisons without confidence intervals.

| K | G | vec4 µs | fused2 µs | tile8 µs | vec4/fused2 | vec4/tile8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 128 | 289.425 | 260.731 | 235.286 | 1.110 | 1.230 |
| 256 | 512 | 269.203 | 245.685 | 216.222 | 1.096 | 1.245 |
| 256 | 2048 | 300.221 | 280.360 | 251.721 | 1.071 | 1.193 |
| 1024 | 128 | 288.508 | 265.120 | 236.360 | 1.088 | 1.221 |
| 1024 | 512 | 289.814 | 266.508 | 245.536 | 1.087 | 1.180 |
| 1024 | 2048 | 866.489 | 845.876 | 831.480 | 1.024 | 1.042 |
| 4096 | 128 | 822.063 | 791.405 | 757.683 | 1.039 | 1.085 |
| 4096 | 512 | 672.962 | 474.971 | 458.813 | 1.417 | 1.467 |
| 4096 | 2048 | 306.045 | 296.675 | 281.527 | 1.032 | 1.087 |
| 16384 | 128 | 677.295 | 666.970 | 497.165 | 1.015 | 1.362 |
| 16384 | 512 | 707.581 | 700.988 | 704.942 | 1.009 | 1.004 |
| 16384 | 2048 | 526.322 | 496.701 | 526.054 | 1.060 | 1.001 |
| 65536 | 128 | 526.443 | 458.971 | 440.508 | 1.147 | 1.195 |
| 65536 | 512 | 479.657 | 465.545 | 483.805 | 1.030 | 0.991 |
| 262144 | 128 | 636.804 | 622.007 | 638.877 | 1.024 | 0.997 |

Tile8 was slightly slower in two admitted panels and almost neutral in others;
all such cases remain in the report. No universal winner, default route,
autotuning, model throughput or model-quality conclusion follows. Repeatability
and review of the external admission evidence are required before performance
promotion. The frozen PVP3b experiment was not run or modified.

## Retained evidence

- [Raw PVP3d rows and samples](evidence/pvp3d-thor-2026-10-06/pvp3d.log).
- [Physical ANF qualification](evidence/pvp3d-thor-2026-10-06/anf-bank.log).
- [ANF executable inventory and identity](evidence/pvp3d-thor-2026-10-06/anf-executable-identity.json)
  and [captured compiler output](evidence/pvp3d-thor-2026-10-06/build.log).
- [Artifact SHA-256 inventory](evidence/pvp3d-thor-2026-10-06/manifest.json).
- [Parsed descriptive summary](evidence/pvp3d-thor-2026-10-06/summary.json).
- Reservation snapshots, lifecycle controller, service states, occupancy and
  restoration evidence are retained in the same directory.

Validation independently recomputed nearest-rank p50/p95 from all thirty
samples per candidate, checked sample indices, complete three-arm cohorts,
checksums, counts and captured artifact hashes. No kernel or dispatch policy
was changed in this evidence-only slice.
