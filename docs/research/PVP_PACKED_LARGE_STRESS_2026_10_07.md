# Packed-address large-domain Thor correctness stress — 2026-10-07

Result: ten of ten fresh Thor processes passed the frozen large-domain
correctness protocol. This is not a speed qualification or model promotion.

## Exact execution identity

- Source and controller: `2065c107f9f32907b21ca944a62aed2ef0d547bd` (PR #345).
- Binary SHA256: `eca57bc3a3a15e4cd326e54f59da06f702cec13ffeb54e75c225fff1a73228ef`.
- Controller SHA256: `0bb5e6d7c1baf544663b9a24a4e31f427a6ff8aad45082eddd1c065bec8c696b`.
- Rust 1.89.0 / LLVM 20.1.7, aarch64; unoptimized test binary, correctness only.
- Adapter: NVIDIA Tegra NVIDIA Thor, Vulkan, installed driver 580.00.
- Reserved: 06:42:06 UTC; completed: 06:47:17 UTC; restored: 06:47:18 UTC.
- Independent restoration inspection: 06:48:16 UTC.

The [prospective protocol](PVP_PACKED_LARGE_STRESS_PROTOCOL.md) and controller
were published before physical execution. The historical comparative benchmark
was neither rerun nor rebuilt. Its SHA256 remained
`7aef67026ad088504478885cf9f2b7c864218f6fb3b0792bd8609fd907d3f414`.

## Functional result

Every process executed five geometries and three bank families, with three
different coefficient rounds per case. The three 4 MiB shapes include
K=16384, K=65536 and K=262144. Two further 129-gate shapes test gate ordering.
Each immutable plan was reused for all uploads, forwards and inverses.

| Quantity | Per process | Ten processes |
|---|---:|---:|
| Geometry/bank cases | 15 | 150 |
| Coefficient rounds | 45 | 450 |
| Forward / inverse checks | 45 / 45 | 450 / 450 |
| Paired source/forward/inverse observations | 135 | 1,350 |
| Complete A/oracle, B/oracle and A/B comparisons | 405 | 4,050 |
| Failed comparisons | 0 | 0 |

Both fresh staging buffers used explicit MAP_READ transitions. The forward
reference is direct monomial-mask evaluation, independently cross-checked
against addresswise evaluation on the original 39 fixtures and all three
rounds (117 host comparisons). Four host tests, native formatting and strict
Clippy passed. No failed GPU process was replaced or omitted.

## Evidence and limits

The [lossless capture](evidence/pvp-packed-large-2026-10-07/capture.tar.gz)
contains all 103 original runtime files, including process outputs, occupancy
records, kernel lifecycle snapshots, initial/restored policies and the final
independent receipt. Its SHA256 is
`d191ca67dbba77626bad824147a548fe0130bb7067e14b6511ac55ddebb42096`.
All manifest byte lengths and SHA256 values were verified after transfer;
the analysis also rejects duplicate/missing comparison keys, incorrect word
counts, wrong source identity, non-Thor adapters and nonzero process exits.

Occupancy observations remained incomplete during runs 1, 3, 4, 6, 7, 8,
9 and 10. These records remain unknown, not clean. The cooperative lock and
sampled observer are not hostile-process GPU fencing. Accepted performance
rows: zero. No GPU latency, gain, spill/register-allocation or model-quality
claim follows from this capture. The earlier automatic-readback faults and
historical regressions remain retained; this does not prove a driver root cause.

All four originally active services and their original restart/manual-start
policies were independently verified restored. Temporary guards, trace
instance and active watchdog were removed, and the device lock was available.

The source execution head had ten of ten observed CI checks green, including
the new mandatory D3D12 WARP large test and full Vulkan/Metal matrices. The
evidence-bearing PR head requires its own green CI before merge; this report
does not substitute the earlier source-head checks for that gate.

Next: implement the [prepared comparison design](PVP_PREPARED_COMPARISON_DESIGN.md)
as a separate frozen measurement harness after this PR merges. Keep final
SML self-sufficiency and the separate future SML-HARNESS boundary unchanged.
