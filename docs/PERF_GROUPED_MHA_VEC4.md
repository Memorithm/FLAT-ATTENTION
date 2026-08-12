# Grouped-forward MHA vec4 candidate

Physical Jetson Thor ranking of FLAT's existing MHA generations identified `Q4Vec4Portable` as the only consistently best existing generation for the measured D=64/128 shapes. The strongest ranking signal was at `seq_len=512, head_dim=128`, where the public-forward median was 2380.105 µs versus 5271.373 µs for `Q4Portable`. That ranking scope included upload and readback, so it is evidence for choosing the next mechanism, not a SciRust release speedup claim.

This candidate reuses the already-qualified M6 vec4 WGSL shader inside `WgpuGroupedForwardPipeline` only when all of the following hold:

- vectorization was explicitly enabled for the pipeline;
- the logical request is MHA (`q_heads == kv_heads`);
- `head_dim` is 64 or 128.

All native GQA/MQA requests and other head dimensions remain on `Q4PortableGrouped`; K/V heads are never expanded. `WgpuGroupedForwardPipeline::new` intentionally remains portable while this candidate is being physically qualified. `with_vectorization(device, true)` enables the candidate explicitly.

Prepared resident requests retain the selected kernel variant, so repeated `encode_prepared` calls use the same pipeline and bindings. Correctness qualification covers D=64/128, causal/non-causal MHA, partial query tiles, output and LSE parity against the scalar grouped oracle. GQA and unqualified dimensions are explicitly checked to remain portable.

Promotion requires a same-context resident Thor benchmark against both the portable grouped path and SciRust's previous multi-dispatch attention. No performance claim is made by this PR alone.

The sovereignty boundary is unchanged: Rust-native host code and WGPU/WGSL only, with no project-authored C/C++ or C ABI bridge and no mandatory CUDA C++/nvcc, WMMA/WGMMA, CUTLASS, cuDNN, or vendor SDK.
