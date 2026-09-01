# FDAL0 — DA-LUC KV audit and research contract

Status: research-only contract audit. This document does not claim DA-LUC novelty, quality, compression, latency, throughput, or runtime promotion.

## Audited repository heads

- FLAT-ATTENTION: `041cef3d9c24e2fdb4c68a75bbb64f5197edb909`
- SLHAv2: `b2cee0d0f30ff0fc752c03193cb9ed93dc91be53`
- NNIS: `a4635547f3e46abd6652f8d319093dabc0baee6f`

These SHAs identify the contracts inspected for FDAL0. They are not dependency pins.

## Current FLAT KV boundary

M14, M15 and M16 already provide the dense reference substrate required before compressed-KV research:

- M14 `WgpuResidentKvCache`: fixed-capacity resident f32 K/V with native `kv_heads`, physical layout `[batch, capacity, kv_heads * head_dim]`, suffix-only device copies, and logical length authority.
- M15 `WgpuResidentDecodePipeline`: `q_len = 1` attention over resident or caller-owned fixed-capacity K/V. It validates geometry, causal visibility, storage binding size and device limits fail-closed.
- M16 `PagedKvTable` / `WgpuPagedKvCache` / `WgpuPagedDecodePipeline`: deterministic logical-to-physical paging and a first single-sequence paged f32 decode path.

None of these contracts identifies a compressed representation. They remain the dense correctness/quality reference and are not changed by FDAL0.

## Cross-repository boundary audit

### SLHAv2

SLHAv2 owns its specialized tile and codec semantics. Its current 128-byte serialized tile carries latent bytes, residual bitmap, scales, dynamic lambda, token id, position, head id, codec flags and group scales. HOT/WARM residency and INT4/NF4/MIXED/TQ3/MIX3 codec details are SLHAv2 product semantics.

FLAT must not copy those flags or tile offsets into its attention contract. A future SLHAv2 adapter may map a compatible SLHA representation into an attention-facing view only when its layout, head/position identity and quality semantics are explicit.

### NNIS

NNIS currently owns a CUDA device-resident generic KV cache with logical shape `[layer][head][capacity][head_dim]`, per-layer live lengths and suffix-only D2D append. Its cached-attention kernel choice is already separated into a versioned, fail-closed execution plan.

FLAT must not copy CUDA pointers, stream ownership or NNIS kernel-plan variants. NNIS can consume a FLAT representation contract only through an adapter that validates the representation before binding NVIDIA-specific storage.

## Prior-art comparison boundary

The purpose of this matrix is to prevent novelty claims from being inferred from known ingredients.

| Work | Relevant KV idea for DA-LUC boundary | FDAL0 implication |
| --- | --- | --- |
| KIVI, arXiv:2402.02750 | 2-bit asymmetric KV quantization; keys and values use different quantization grouping directions | K/V asymmetry is established prior art; it must be explicit metadata, not a novelty claim |
| KVQuant, arXiv:2401.18079 | per-channel/pre-RoPE key treatment, non-uniform low-bit quantization, dense-and-sparse outlier handling | outlier-preserving sparse residuals and asymmetric K handling are established ingredients |
| QServe, arXiv:2405.04532 | W4A8KV4 serving co-design, SmoothAttention and fused attention around a low-bit KV cache | a low-bit cache consumed by an attention kernel is not sufficient for a novelty claim |
| PQCache, arXiv:2407.12820 | product quantization of keys for MIPS-based important-token retrieval using PQ codes/centroids | PQ/codebooks and lookup-based key search are established; DA-LUC must be compared against this boundary rather than renaming it |
| TensorRT-LLM current quantization/KV documentation | paged KV plus INT8/FP8 and current NVFP4 KV-cache modes | quantized paged KV is an external production baseline when hardware/model regimes are reproducibly comparable |

Primary references:

- https://arxiv.org/abs/2402.02750
- https://arxiv.org/abs/2401.18079
- https://arxiv.org/abs/2405.04532
- https://arxiv.org/abs/2407.12820
- https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html
- https://nvidia.github.io/TensorRT-LLM/features/kvcache.html

No statement above establishes novelty for a combination of these ingredients. A future novelty statement requires a dedicated literature audit beyond this implementation-boundary matrix.

## FDAL0 v1 contract

`flat_attention::research_da_luc` is intentionally outside the stable `api::v1` namespace. It describes one per-layer attention-facing KV view and nothing more.

The v1 descriptor makes these facts explicit:

- logical batch, query-head count, KV-head count, live KV length, K feature width and V feature width;
- exact GQA/MQA mapping requirement: `q_heads % kv_heads == 0`;
- uniform K subspace width;
- K codebook entry count, codebook floating dtype and shared/per-KV-head scope;
- packed K index width and bit order;
- independent V representation kind: dense or groupwise affine low-bit;
- V group size, scale dtype and zero-point storage when groupwise quantized;
- optional sparse K/V residual semantics with coordinate or bitmap indexing;
- contiguous or paged physical capacity metadata;
- row order, per-plane alignment and padding rule.

The descriptor has no free-form codec name and no private repository flag. Adapters must map their own representation to these semantics explicitly.

## Fail-closed rules in v1

Validation rejects:

- unknown schema versions;
- zero dimensions;
- ambiguous/non-integral query-head to KV-head grouping;
- K subspace widths that do not partition the key dimension;
- packed index widths unable to address the declared codebook;
- codebook indices outside the declared entry count;
- V group sizes that do not partition the value dimension;
- zero-point storage too small for the declared V value width;
- malformed sparse residual budgets or coordinate widths;
- residual coordinates outside the logical vector;
- physical capacity below the live KV length;
- non-power-of-two or zero plane alignment.

## Explicit non-goals

FDAL0 does **not** provide:

- an encoder or decoder;
- a CUDA/WGPU compressed-attention kernel;
- a codebook trainer or calibration algorithm;
- a physical page table payload format;
- exact effective-bits-per-value accounting (FDAL1 requirement);
- quality or real-model evidence;
- a runtime default, fallback or routing policy;
- PHR-Lite routing;
- a claim of "zero dequantization";
- an 8–16x memory claim.

The verifiable future kernel property is **no dense K/V materialization**, not the absence of scalar/register conversion.

## Next gate

FDAL1 may begin only from a validated v1 descriptor plus the unchanged dense reference path. It must add a host reference encode/decode and exact storage accounting that includes codebooks, scales/zero-points, residual values and indices/bitmaps, metadata, alignment and padding before any compression ratio is reported.
