# Thor repetitions and Dell cross-device diagnosis — 6–7 October 2026

## Result

**Gain confirmation is rejected.** The explicitly requested ten-process Thor
battery completed: **9/10 processes passed the whole-state oracle; one failed
with 16,320 incorrect u32 words**, before timing K=262144, G=128, vec4.
One measuring-phase occupancy snapshot was also unknown. Both conditions
independently exclude this battery from confirmatory performance evidence.

The preceding three Thor reruns and three native Dell reruns each passed.
Passing a smaller series does not erase the later reproducible intermittent
failure. No model-quality, runtime-selection, default-route or autotuning
promotion is made.

## Latest requested executions

| Campaign | Device | Processes | Exact complete processes | Raw diagnostic samples | Confirmation |
|---|---|---:|---:|---:|---|
| R8, three reruns | Thor, Vulkan 580.00 | 3 | 3 | 4050 | Insufficient repeatability / previous failures open |
| Dell D2 | Dell T430, RTX 4060, Vulkan 595.45.04 | 3 | 3 | 4050 | Shared device users; functional diagnosis only |
| R11, ten tests | Thor, Vulkan 580.00 | 10 | 9 | 13410 | Rejected: exact oracle failure and unknown occupancy |

R11 used two independently acquired/released reservations of five processes.
Its 447 measured diagnostic rows include 42 rows from the failed process;
165 limit-rejected rows are retained. Nine completed processes contribute
12150 samples, the incomplete failed process another 1260. **Accepted
confirmation rows: zero.** Do not compute a winner by dropping the failure.

R11 also passed 192 ANF candidate cases across its two blocks. R8 and Dell D2
each passed 96 ANF cases. These 24-case synthetic ANF checks and coefficient
recovery do not cover the complete large-width benchmark or model semantics.

The frozen ten-process engineering rule required all ten complete exact
processes, accepted occupancy, and per-panel >=5% p50 reduction in >=9/10
processes plus >=5% reduction in each reservation-block median. The prerequisite
failed, so no such gain is accepted. This rule is not a significance test or
confidence interval. R1–R7's earlier fifteen-process criteria remain unchanged.

## Failed campaigns remain part of the record

| Campaign | Prospective change | Retained outcome |
|---|---|---|
| R1 | Original source, complete scans only | First invocation aborted on partial occupancy |
| R2 | Detailed v2 gaps, verified owned fd-metadata churn | Three full processes; fourth aborted on unidentified exited PIDs |
| R3 setup | Kernel lifecycle ancestry | Controller PID quoting error; no ANF/timing; services later explicitly repaired |
| R3 | Corrected controller / trace ancestry | Vec4 K262144 G128: 16864 wrong words |
| D1 | Shader-free changing 4 MiB copy matrix | Two non-flushed retained-readback rounds failed; admission incomplete |
| R4 | RustDesk additionally paused | Vec4 K262144 G128: 160 wrong words |
| R5 | Explicit initial-upload completion | Vec4 K65536 G512: 272 wrong words |
| R6 | Retain completed readbacks | Tile8 K16384 G2048: 41080 wrong words |
| R7 | Also retain source/state buffers | Fused2 K16384 G2048: 2384 wrong words |
| R9 setup | Requested ten-process controller | Incorrect source path; failed before reservation/timing |
| R9 | Two blocks of five | Four full processes; fifth aborted on unknown occupancy |
| R10 | No auxiliary shell/SSH during timing | Two full processes; third aborted on unknown occupancy |
| R11 | Complete ten functional tests; unknown scans exclude performance | Nine pass; one vec4 failure, 16320 wrong words |

Every protocol was published before its corresponding execution. None of these
failures was silently replaced inside its campaign. R11 deliberately completed
the requested ten functional processes while retaining failed or unknown
performance-admission states.

In R3, R4 and R5, every recorded wrong word equals the immediately preceding
geometry's scalar-oracle word at the same index. This is verified against the
pinned Rust crate, and establishes stale content in the returned data; it does
**not** identify whether upload, copy, execution, mapping, lifetime, backend or
driver caused it. R6/R7 do not share that immediate-predecessor pattern.
The eight reported R11 wrong-word values repeat the corresponding R3 values;
the other R11 wrong words are counted but not printed.

D1 had no compute shader: six cells of 30 changing 4 MiB transfers each.
Non-flushed drop/reuse cells passed, non-flushed retain failed 2/30; all three
explicit-flush cells passed. Its shared/unknown observation prevents exclusive
qualification. This diagnostic does not establish that flushing fixes PVP:
R5's full benchmark failed despite the flush.

## Source and device identity

Original R1–R4 benchmark:
`50c0230cc645bd44f1520e2c7194878a4b13009c`, binary SHA256
`f4b7f2a4e3d31147c80c2cb724a981c48f6c402002dbb4bd036f2d6a93083d81`.

R5, R6 and R7 experimental source commits respectively:
`64106a9afeab7d3ac079546a4dc0173fb62bacf7`,
`7adae9ae79f7c96e606a05abec95b7b9aee1c196`,
`d7eef77562dc952176bc67c9e5c440ac92459a10`.
Their clean source/build identities and prospective protocols are retained.

R8–R11 and Dell D2 all execute source
`ddb6f0144821dfa29d268601e7087e5001278e52`.
This reverts the ineffective lifetime/fence experiments and only bounds exact
oracle failure reporting. The kernels, fixture, grid, balanced order, 5 warmups,
30 repetitions and resident timing interval are unchanged. Comparison work
outside the timer now counts all mismatches and retains the first eight.

Native Rust 1.89.0 builds:
- Thor benchmark SHA256:
  `7aef67026ad088504478885cf9f2b7c864218f6fb3b0792bd8609fd907d3f414`;
- Dell benchmark SHA256:
  `531740ef18ab2ff447ea6c022f25f89ae897b649029e5d36238a5f0105a89a2f`.

Thor adapter is NVIDIA Tegra NVIDIA Thor / Vulkan / IntegratedGpu / 580.00.
Dell is PowerEdge T430 / x86_64 / NVIDIA GeForce RTX 4060 / DiscreteGpu /
Vulkan / 595.45.04. Dell's ANF and benchmark were compiled natively; its other
observed device users were not interrupted. Samples are never pooled across
these devices.

The Rust std-only observer is pinned to RemoteOps PR268 source
`7cc8203ea9c69349fe19198efe8d04784165d8bc`, using the recorded supporting
workspace at `a6c6c6c2e7ba082a957902d0aab18ba01c0672a5`.
Thor observer SHA256 is
`2750a44e6167875081368cd7df6b7ce582e77df17473ce5685727659035149fd`;
native Dell observer SHA256 is
`b0bb9662863324e244cfec3e6807acc7c33a1ad0097d79ab7856830b9b8cc775`.
RemoteOps PR269's bounded indexed aggregation is merged separately; the
measurement observer was not replaced midway.

## Reservation and restoration

CLM, CLM encoder and Viggle were stopped with temporary runtime manual-start
guards and Restart=no. R4 onward also paused RustDesk, whose short-lived
children had caused scan gaps; Remote Desktop Commander is an independent
service. An independently contending cooperative device flock, owned-task
process checks and prearmed independent restoration timers were retained.

Partial observations never become observed or GPU-idle/exclusive claims.
Verified owned metadata churn or trace-confirmed owned exits are qualified by
a separate controller decision; truncated traces, gaps of unknown ancestry,
limits or foreign users fail closed. Scheduler traces record numeric lifecycle
events in a task-specific instance, with zero-loss checks; this is not hostile
process fencing or proof against cgroup migration.

A failed R3 setup exposed the encoder's start-limit restriction during repeated
restoration. It was reset and the services explicitly restarted; later restore
scripts reset failed/start-limit state before restoring unchanged policies.
R8 has a stale `block_complete=5_invocations` literal in its unmodified log:
its frozen loop, timestamps and exit records prove exactly three processes.
The summary records this erratum; the raw controller/log are not rewritten.

Final inspection at 2026-10-06 22:42:56 UTC found all four Thor services active,
original policies restored (three on-failure, RustDesk no), manual guards and
trace instances absent, restoration timers inactive and the device lock free.
Dell's bounded diagnostic unit exited 0 and its cooperative device lock is free.

## Artifact and implementation scope

Evidence is in [the archive](evidence/pvp3d-thor-repeat-2026-10-06/).
Controllers, checker and analysis sources are archived as audit text, not a new
model/runtime dependency. Original R3–R7 panic logs each contain two million-word
vectors; compact logs substitute those two lines. Their scalar oracle plus
`mismatch-delta.csv` reconstruct both arrays losslessly, checked against packed
little-endian u32 SHA256 values and original-log hashes in each manifest.
Original oversized logs remain on Thor. R11 uses bounded native failure output.

Final implementation retains original benchmark buffer lifetimes and emits
compact exact failures. A separate portable shader-free 40-round changing-upload
test validates every copied word with explicit upload completion and bounded
160 MiB retained readbacks. It passed on Thor in the experimental builds, but
**is not a fix for the large-width PVP failure** and cannot replace its oracle.
The final test discovers the platform's WGPU backend; CI validates the exact
final source independently.

Next qualification gate: isolate the failing transfer/execution/readback stage
under a minimal large-width reproducer and obtain complete controlled
observations, then rerun a new prospectively frozen battery. SML keeps model
semantics and self-sufficiency; no mandatory FLAT/SciRust/NNIS runtime dependency
or vendor SDK above the installed GPU driver is introduced.
