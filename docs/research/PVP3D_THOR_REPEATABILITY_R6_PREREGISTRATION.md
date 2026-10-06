# PVP3d-R6 — corrected upload completion and quiet Thor repetitions
Frozen before any R6 execution. R1/R2 admission failures, R3/R4/R5 correctness
failures and D1 diagnostics are excluded from every R6 gain decision.

## Execution identity and fixed procedure
- Clean source: 7adae9ae79f7c96e606a05abec95b7b9aee1c196.
- This source adds an explicit queue submission/completion after initial upload,
  before resident reset. It is outside warmup/measured intervals. Candidate
  kernels, geometry, order, reset/readback scopes and timing boundaries are
  unchanged. R5 passed its 40-round transfer and 96-case ANF gates but still
  failed the complete benchmark, so the upload fence alone is not a fix.
- R6 additionally retains completed oracle-readback buffers until process exit,
  preventing allocation reuse across geometries. Retention and initial upload
  completion remain outside the timer; the retention residency is an explicit
  changed harness condition, not a new kernel. Record count and payload bytes
  separately from the four-state payload and never as total VRAM. On the known
  15 admitted Thor geometries: 90 buffers, 100491264 payload bytes.
  Do not claim a driver root cause or a validated fix before execution.
- Retain build output, explicit executable paths and SHA-256 identities before
  every block and after it. Use the newly built ANF and transfer-test binaries.
- Three acquired/released blocks, five fresh benchmark processes per block:
  15 complete invocations. No extra run chosen from observed latency.
- Each invocation: full frozen PVP3d grid, all three arms, five warmups,
  thirty repetitions, balanced six orders, full CPU oracle before and after.
  Common-limit rejection preserves complete cohorts; retain all raw samples.
- Once per block, first run the 40-round 4 MiB shader-free transfer regression
  (every word must match), then the 24-case/four-kernel ANF suite. Any correctness
  failure aborts the campaign; do not replace a failed block.
- Cooperatively lock /dev/nvidia0, verify independent contention; pause and
  guard CLM, encoder, Viggle and RustDesk. Desktop Commander is a different
  service and stays available. Prearm an independent restoration timer.
  Restore all four services between blocks, remove only this block's guards,
  reset their systemd start-limit counters when needed without changing policy.
- RemoteOps v2 observer at 7cc8203ea9c69349fe19198efe8d04784165d8bc:
  limits/truncation/foreign users remain unqualified. Partial metadata gaps may
  be attributed only to verified owned process identity. Exited fd_directory
  gaps require owned fork/exit ancestry with PID reuse invalidation, captured
  kernel trace snapshot and zero overrun/loss on every CPU. No unknown or
  foreign gap is accepted during execution; retry only outside measurement.
  Retain partial status explicitly, not as an observed/exclusive capability.
- Separate tracefs instance, global clock, 4096 KiB per CPU, numeric fork/exit
  records only. Cleanup disables/removes only this instance. This is trusted
  descendant attribution, not hostile-code fencing or a GPU utilization meter.

## Unchanged prospective descriptive decision
For each admitted geometry and each of fused2/tile8, compute invocation latency
reduction: 1 - candidate_p50 / vec4_p50. Report all fifteen values, min/median/max,
each block median and candidate p50/p95. Accept a repeatable resident-wall gain
in R6 only if all fifteen invocations and three blocks are valid, at least 14/15
reductions are >= 0.05, and every block median is >= 0.05.
Report all thirty decisions, including rejected gain criteria and six
limit-rejected cohorts. The original capture is never pooled into R6.
This is a fixed engineering rule, not a p-value, confidence interval or claim
that samples/processes/blocks are statistically independent. One device/day
and four paused services delimit the result. No universal winner, default
routing, autotuning, SML quality/training or model-throughput promotion.

## Validation
Recompute nearest-rank p50/p95 from raw samples, require indices 0..29 per arm,
complete three-arm cohorts and matching input/output checksums across runs.
Require source identity, hardware Vulkan classification and completion markers.
Preserve failed campaigns with exact raw-log hashes and lossless mismatch deltas;
the two 24 MiB vector dumps can be reconstructed from the deterministic CPU
oracle plus actual differing words and verified packed-vector SHA-256 hashes.
