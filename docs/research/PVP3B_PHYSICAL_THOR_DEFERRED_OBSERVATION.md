# PVP3b physical Thor observation — deferred GPU-busy attempt

Status: retained operational evidence. No PVP3b performance result was accepted.

## Identity

- Programme: PVP3b fused2 same-device benchmark.
- Control: merged PVP2 vec4, one butterfly stage per dispatch.
- Candidate: merged PVP3a fused2, first two stages fused in registers.
- Benchmark implementation merge: `26690c7c467db24ae5fc3bebe235c1f0310f4f2a`.
- PR: #337.
- Qualification workflow run: `37513743739`.
- Physical job: `112441232837`.
- Runner class: persistent Jetson Thor runner required by
  `.github/workflows/pvp3b-thor-qualification.yml`.

## Frozen protocol

The attempted run used the preregistered protocol:

- K = 256, 1024, 4096, 16384, 65536, 262144, 1048576;
- G = 128, 512, 2048;
- five warmups;
- thirty measured repetitions;
- alternating PVP2/PVP3a candidate order;
- identical resident initial state;
- exact readback parity before timing;
- state reset completed outside the primary timer;
- primary timing scope = encode + queue submit + device wait;
- optional GPU timestamps reported separately;
- no CUDA/NVRTC/cuDNN/TensorRT path.

## Observation

The physical Thor workflow completed successfully but **deferred measurement**
before the benchmark timing loop because the GPU was already occupied.

The retained workflow output was:

```text
qualification_status=deferred_gpu_busy
performance_claim=none
```

No `measured` or `skipped_limit` benchmark rows were accepted from this
attempt.

## Interpretation

This is not a negative performance result for PVP3a and not a positive result
for PVP2. It is an operationally blocked attempt.

The workflow behaved correctly:

1. exact source and persistent Thor identity were checked;
2. the benchmark binary was built before GPU reservation;
3. the physical driver device was used as an exclusive lock target;
4. pre-existing GPU compute occupancy was detected during the settle period;
5. the run exited without publishing timing data.

This retained failure mode is important because accepting measurements during
foreign GPU activity would invalidate the same-device comparison.

## Next action

Re-run the **unchanged** PVP3b protocol on a clean Thor GPU. Do not change the
K/G grid, warmups, repetition count, candidate order, timing scope, or decision
rule in response to this deferred attempt.

Only a future run that emits the complete frozen set of benchmark rows is
eligible for device-specific PVP3b performance interpretation.
