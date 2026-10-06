# PVP3d-R2 — Thor repeatability protocol

Frozen prospectively before the first R2 measurement. R1 was aborted after an
unattributed partial scan during its first invocation; all R1 timings are
excluded. R2 changes only the external gap-attribution policy, not kernels,
geometry, warmups, repetition counts, order or the gain acceptance rule. The earlier PR342 capture
is exploratory context and is excluded from the R2 decision dataset.

## Fixed execution

- Reuse the unchanged clean source 50c0230cc645bd44f1520e2c7194878a4b13009c.
- Benchmark SHA-256: f4b7f2a4e3d31147c80c2cb724a981c48f6c402002dbb4bd036f2d6a93083d81.
- ANF executable: target/release/deps/pvp_anf_bank-f2f5ea0ec5889510,
  SHA-256 4458f8e245476cb5ec65ace1b8ab57f09bc7a9e4e137a4e70328b9652987adfa.
- Verify both hashes before each reservation and again after each block.
  Use the explicit ANF executable path, never find/head selection.
- Three separately acquired/released reservations, five fresh benchmark
  processes per block: 15 invocations total. Restore the original three GPU
  services between blocks. No extra runs based on observed latency.
- Each invocation uses the entire unchanged PVP3d grid: 21 geometries,
  all three arms, five warmups, 30 repetitions, balanced six orders,
  full CPU oracle before/after. Preserve common-limit rejections.
- Run the physical 24-case/four-kernel ANF suite once per block.
- Device-user observer: RemoteOps opt-in v2 --details at source
  7cc8203ea9c69349fe19198efe8d04784165d8bc. The observer still emits partial
  status and denies both idle/exclusivity capabilities. Retain its exact binary
  hash and pinned materialization context separately.
- Admission remains operator-controlled cooperative flock, paused CLM/encoder/
  Viggle services, runtime start/restart guards and prearmed restoration timer.
  Require observed device-user scans attributed to the qualification cgroup.
  The external checker can accept a partial scan ONLY when every gap is an
  fd_metadata gap of a PID whose process identity is verified and whose cgroup
  is the exact qualification unit. The sum of gap counts must equal all unreadable
  entries. Record each such observation as partial_owned_descriptor_churn, never
  as observed. A limit, truncated details, missing/changed identity, directory or
  enumeration gap, foreign visible user or foreign gap is unqualified.
  Outside measured execution retry unqualified scans at most five times.
  During measurement allow no retry of an unqualified scan: reject the block.
  Abort the campaign on a rejected block; preserve it and do not replace it.
- Retain phase/time-tagged occupancy, exact source/binary identity, process logs,
  raw samples, original/restored service state and host load.
- No CUDA, NVML, nvidia-smi, vendor SDK or new model runtime dependency.

## Prospective decision rule

For each admitted (K,G) and each candidate (fused2 or tile8), compute each
invocation's latency reduction as 1 - candidate_p50 / vec4_p50.
Report all 15 values, median/min/max, candidate p50/p95 and each block median.
Call a geometry's resident-wall gain repeatable in this campaign only if:
1. all 15 complete invocations and all three blocks are valid;
2. at least 14/15 invocation reductions are >= 0.05;
3. each of the three block median reductions is >= 0.05.

This is a fixed descriptive engineering acceptance rule, not a hypothesis test,
confidence interval or statistical independence claim. Processes in a block
share device state and the three blocks share one device/day. Do not pool
20,250 raw samples as independent trials or claim population significance.
All 30 geometry/candidate decisions, including failures, are reported.
A failed criterion means gain not confirmed by this campaign, not proof of
equal performance. No universal-winner, default-route, SML model-quality,
training, bandwidth or model-throughput promotion follows.

## Validation

Require per invocation 63 candidate rows (complete three-arm cohorts), each
measured arm's raw sample indices exactly 0..29, nearest-rank p50/p95 matching
its samples, matching input/output checksums across arms and invocations,
hardware classification and completion marker. Report actual measured/limit
counts; do not assume a rejected geometry has measured data.
Retain raw artifact SHA-256 inventory and the controller/checker as audit text.
External admission review remains open: sampled visibility and cooperative
locking do not provide hostile-process fencing or hardware utilization.
