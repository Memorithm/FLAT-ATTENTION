# Prepared resident-wall diagnostic: prospective protocol
Frozen cohort: Thor Vulkan only; release Rust 1.89.0, five already-qualified prepared arms.
Six fixed (K,G): (4096,128), (16384,2048), (65536,512), (262144,128), (16384,129), (65536,129).
Three ANF banks: boundary, sparse4, dense32; nonce rounds 0,1,2 cycle across trials.
Three fresh processes, five warmups/arm/case, twenty repeats/arm/case.
Measured order: ten cyclic/mirrored permutations repeated twice, equal positions and pair precedence.
Common capability rejection excludes a shape for all five arms, with every reason retained.
Primary forward wall timer: start before encoder creation; stop after submit and explicit device completion.
Inverse wall is a separate metric. Upload/reset is independently timed and fenced outside resident wall.
Paired explicit MAP_READ source, forward and inverse checks include every canonical word, padding and four guard words.
Every timed state is verified outside its timer. Failed samples and processes are retained; no rerun substitutes failures.
Pipeline/state-buffer/immutable-plan preparation and physical source/oracle conversion have separate records.
End-to-end latency and GPU timestamps are NOT measured in this first diagnostic cohort; timestamps are not_requested, never unsupported by inference.
No model, physical-register, bandwidth, universal-winner or default-route claim. Raw resident-wall ratios are descriptive only.
Occupancy unknown remains unknown. Independent performance admission remains open; accepted confirmation rows zero.
Three fresh processes produce 3600 measured forward/inverse samples/process (10800 total when all shapes admitted), plus 900 warmup samples/process.
Use existing scoped four-service guard, independent 30-minute restore watchdog, cooperative GPU flock, frozen Rust device-user observer and 25-minute unit budget.
Per-process timeout 360 seconds. Freeze formatted source/controller and copied binary SHA before native execution.
Never rebuild or overwrite the frozen historical native PVP3d executable. Publish all logs and independent service restoration inspection.
