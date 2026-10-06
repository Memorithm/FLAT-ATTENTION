# Pascal Vector Projection (PVP) — FLAT hardware bootstrap

Status: research-only hardware realization programme.

## Scope

FLAT-ATTENTION participates in PVP as the portable GPU kernel laboratory for
SML-GENIUS Pascal/subset-zeta and massive ANF-bank execution. It does not own
the model semantics and it must not become a mandatory dependency of the final
SML model.

The semantic owner is SML-GENIUS. SciRust owns reusable bitplane/SIMD
primitives. NNIS may provide optional portable runtime/session qualification.

## Hardware contract

The hot representation is a regular bit-sliced matrix, not a graph:

- logical: address-major gate bitplanes;
- portable GPU word: `u32`;
- vector transaction candidate: `vec4<u32>` = 128 gate bits per address;
- inner gate-word dimension contiguous;
- transform: deterministic subset-zeta/Pascal XOR butterfly.

No adjacency list, node object, pointer chasing or sparse graph traversal is
required for the Pascal lattice hot path.

## Sovereignty

For this programme, portable CPU/WGPU/open-GPU execution is authoritative.

The only vendor-specific software layer admitted in the target path is the
installed GPU driver required by the open backend. Do not introduce a required
dependency on CUDA, NVRTC, cuDNN, TensorRT/TensorRT-LLM, NVML, CUTLASS, CUBIN
runtime contracts, or project-authored C/C++ vendor-SDK bridges.

Existing NVIDIA-specific code or historical evidence in other repositories is
not PVP qualification evidence.

## FLAT primitives to reuse

- `vec4` memory transaction machinery;
- workgroup tiling and shared staging;
- subgroup capability detection and deterministic fallback;
- double-buffer candidate structure;
- Kernel IR separation of semantic problem and realization;
- capability prefiltering;
- correctness-before-timing candidate qualification;
- benchmark/evidence manifests;
- Boolean Front-End M13B principle: make compact Boolean decisions before
  expensive numerical staging.

PVP requires a new Boolean-butterfly kernel family. Do not overload
`DenseQ4Forward` or pretend its f32 configuration types describe PVP.

## Programme

1. FLAT-PVP0: consume the frozen versioned bitplane layout and implement a host
   adapter/oracle differential.
2. FLAT-PVP1: scalar WGSL `u32` butterfly reference.
3. FLAT-PVP2: `vec4<u32>` vectorized loads/stores across gate bitplanes.
4. FLAT-PVP3: fuse multiple butterfly stages inside one workgroup when resource
   and barrier constraints permit.
5. FLAT-PVP4: evaluate subgroup-assisted exchange/reduction only where it maps
   to the XOR network; retain deterministic fallback.
6. FLAT-PVP5: evaluate single/double-buffer staging and tile geometries.
7. FLAT-PVP6: deterministic candidate generation, capability filtering,
   correctness qualification, then real-device timing/autotuning.
8. FLAT-PVP7: export only a minimal versioned PVP kernel/layout contract to
   SML-GENIUS and optionally NNIS portable sessions.

## Promotion boundary

A faster PVP kernel does not alter default dense attention, M13B routing, SML
architecture or model quality claims. SML decides whether to internalize a
qualified realization after destination requalification.
