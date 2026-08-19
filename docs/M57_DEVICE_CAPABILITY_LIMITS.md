# M57 — Phase I device capability limits

M57 continues the roadmap's Phase I autotuning work (ROADMAP M24) from verified `main` commit `48130e876dfebcfd094a6f9ae5866d6efa95d126`.

M56 added the deterministic device identity fingerprint. This slice adds the explicit resource/capability limits that M24 requires before candidate generation and pipeline creation.

## This slice

`RuntimeDeviceCapabilities` now provides a deterministic host-side model of:

- maximum workgroups per dimension;
- maximum workgroup size on each axis;
- maximum workgroup storage bytes;
- maximum bind-group count (wgpu 0.20 `max_bind_groups`);
- maximum storage-buffer binding size;
- subgroup support and subgroup size range;
- f16 (`SHADER_F16`) support.

The record:

- serializes deterministically through `canonical_record()`;
- produces a stable FNV-1a-64 fingerprint through `stable_fingerprint()`;
- is captured once at WGPU context creation from the actual selected device limits/features;
- is exposed through `WgpuFlatAttention::device_capabilities()`.

Adapter marketing names remain provenance only. Candidate selection must use these explicit limits, not name heuristics.

## M24 boundary

M24 requires the capability model to serialize deterministically and to filter unsupported configurations before pipeline creation. This slice closes the first half (deterministic serialization) and provides the host-side limit surface. Wiring the limits into each pipeline's pre-creation candidate filter (workgroup size/storage and binding validation per candidate) is the next Phase I slice. M25 deterministic candidate generation and M26 benchmark-driven selection remain later work.

No WGSL, kernel routing, numerical behavior, `api::v1` contract, benchmark result or performance claim changes in this PR.
