# Dell D2 — three cross-device diagnostic reruns

Frozen before execution on the user's accessible Dell T430 (`debian`, x86_64) following the 2026-10-07 Europe/Paris request. Known SSH host identity and existing authorized key are reused through Thor. No credential copying, host-key bypass or vendor-runtime tools.

Use exact FLAT source ddb6f0144821dfa29d268601e7087e5001278e52, Rust 1.89, WGPU Vulkan, the frozen PVP3d 21-geometry grid and original buffer lifetimes with bounded exact-oracle diagnostics. Build native binaries; run explicit ANF test executable, then three fresh benchmark processes, retaining failures and whole-state exact comparison. Record compiler/source/binary hashes and actual adapter/backend/driver. Limit-rejected cohorts remain recorded.

Acquire an independently contending cooperative /dev/nvidia0 lock with a bounded supervised task. Record Linux visible-device-user snapshots using the pinned RemoteOps v2 Rust observer; detected descriptors are not compute-activity or hostile-fencing evidence. Do not interrupt unrelated Dell jobs. If other GPU users or unknown scan gaps remain, runs are functional/diagnostic only, and wall samples are excluded from gain qualification. GPU tests may still establish exact correctness under the observed shared condition. A lost cooperative lease aborts execution.

This is cross-device diagnosis, not a confirmatory 15-process campaign or a model benchmark. No gain/default/runtime/autotuning promotion; do not combine Dell and Thor latency samples. Preserve all three exits and services/lock cleanup.
