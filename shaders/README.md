# Handwritten WGSL shaders

These sources are the portable GPU kernels. Generated Kernel-IR emission lives
in `src/kernel_wgsl.rs` and is **not** selected by production routing until a
milestone document says otherwise.

Presence in this directory is not a performance claim and is not automatic
default routing. Default dense forward remains the qualified M4 Q4 path
(`flat_fwd.wgsl`) unless a capability-gated policy selects another qualified
variant.

## Qualified / default-family dense forward

| File | Role |
|------|------|
| `flat_fwd.wgsl` | M4 Q4 portable fused forward (qualified default). |
| `flat_fwd_single.wgsl` | M2/M3 one-query-row baseline, retained for comparison. |
| `flat_fwd_subgroup.wgsl` | M5 subgroup-assisted Q4; selected only when `Features::SUBGROUP` is present and policy allows it. |
| `flat_fwd_vec4.wgsl` | M6 vectorized storage for supported head dims. |
| `flat_fwd_double_buffer.wgsl` | M7 double-buffered staging; opt-in, evidence-gated. |
| `flat_fwd_f16.wgsl` | M8 packed-binary16 I/O with FP32 accumulation. |

## Grouped / RoPE / projection layouts

| File | Role |
|------|------|
| `flat_fwd_grouped.wgsl` | M10 native GQA/MQA without physical KV-head expansion. |
| `flat_fwd_grouped_vec4.wgsl` | Vectorized grouped forward. |
| `flat_fwd_grouped_rope.wgsl` | FLAT-R1 head-local RoPE fused into Q/K staging. |
| `flat_fwd_projection_rope.wgsl` | FLAT-R2 sequence-major projection + RoPE + GQA. |
| `flat_fwd_projection_rope_asymmetric.wgsl` | M11 rectangular Q/KV lengths. |
| `flat_fwd_projection_rope_asymmetric_vec4.wgsl` | M53 opt-in vec4 loads on the rectangular path. |
| `flat_fwd_projection_rope_rect.wgsl` | Rectangular projection-layout variant. |
| `flat_fwd_projection_rope_variable.wgsl` | M12 padded variable-length batches. |
| `flat_fwd_q1_vec4.wgsl` | M58 opt-in Q1 vec4 MHA (one workgroup per query row). |
| `flat_fwd_q1_direct_vec4.wgsl` | M60 Q1 direct vec4 candidate. |

## Decode and training

| File | Role |
|------|------|
| `flat_decode_resident.wgsl` | M15 `query_len=1` decode over fixed-capacity resident KV. |
| `flat_decode_paged.wgsl` | M16 decode over paged resident KV. |
| `flat_decode_projection_kv_reuse.wgsl` | Decode that reuses already-projected/rotated K. |
| `flat_backward_recompute.wgsl` | M18 correctness-first backward recomputation. |
| `flat_backward_grouped_recompute.wgsl` | M19 grouped GQA/MQA backward recomputation. |

## Research-only

| File | Role |
|------|------|
| `flat_da_luc_decode.wgsl` | DA-LUC research decode candidate. Not `api::v1`. No default routing. |

See [`docs/SHADERS.md`](../docs/SHADERS.md) and [`docs/m9-numerical-policy.md`](../docs/m9-numerical-policy.md).
