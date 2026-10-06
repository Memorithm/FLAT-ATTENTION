# PVP readback isolation — prospective diagnostic protocol, 7 October 2026

This is a new correctness-only experiment. Historical PVP3d source, timings,
failures and admission decisions are not modified or replaced.

## Question and primary source

Can two independent copies/readbacks of an unchanged GPU buffer disagree,
and does an explicit COPY_DST-to-MAP_READ transition change the outcome?

The installed wgpu-core 30.0.1 `resource.rs` map_async path references
[wgpu issue 9306](https://github.com/gfx-rs/wgpu/issues/9306), opened by
andyleiserson on 25 March 2026, about a potentially missing pre-mapping
barrier. That issue is a hypothesis source, not a diagnosis of Thor.
`CommandEncoder::transition_resources` is the safe, native WGPU interface;
it is documented as a no-op on web. No vendor API or dependency patch is used.

## Frozen experiment

- Two modes in one pinned source/binary: `--diagnose` (automatic transition)
  and `--diagnose-map-transition` (explicit transition of both readbacks after
  their copies and before submission).
- Six fresh processes in order automatic, explicit, automatic, explicit,
  automatic, explicit. No failed process is replaced.
- Twelve rounds per process. Each round visits K=65536/G=512 and then
  K=262144/G=128, which both occupy 4 MiB in the historical vec4 gate layout.
- Deterministic historical fixture and CPU subset-zeta oracle.
- Source upload, reset state and transformed state are checked independently.
  All three historical candidates are retained. Each observation copies the
  same unchanged source into two distinct readback buffers in one submission.
- Every returned word is compared; retain full mismatch count, first eight
  tuples, checksum and disagreement count between paired readbacks. Continue
  bounded diagnostics after data mismatches, then fail the process. WGPU/API
  errors abort and remain explicit failures.
- No timer-derived performance claim; no small stress result qualifies R11.
- Record exact source SHA at compile and run time, binary SHA256, backend,
  adapter, driver, timestamps and return code.

## Host control

Use a separate cooperative device reservation on Thor, a second contending
lock check, initially captured service policies, temporary runtime guards and
an independent restoration timer. Pause the previously authorized CLM,
encoder, Viggle and RustDesk services only after confirming the control channel
is independent. Restore and verify their initial states/policies and release
the guards, traces, timers and device lock on every exit.

Retain RemoteOps device-user observations before, during and after the process.
Unknown/partial observations remain unknown/partial. This experiment may
localize a correctness defect under diagnostic admission; it cannot establish
GPU-idle or hostile-process exclusivity and yields no accepted timing rows.

## Interpretation fixed before execution

- A/B readback disagreement from one unchanged source localizes the observed
  discrepancy to the copies/readback path, not exclusively to the transform.
- Two readbacks matching the same wrong words do not distinguish incorrect
  source/state from shared readback/coherence failure.
- Fewer failures with an explicit transition support that control as a
  candidate workaround, but do not identify the driver as root cause or
  prove an intermittent problem eliminated.
- Six successful processes cannot erase earlier failures. Re-run the full
  historical geometry/order stress and separately qualify performance only
  after a discriminating correctness investigation.

The separate packed-address candidate is a layout experiment, not a fix for
readback visibility, and is excluded from this paired-readback protocol.
