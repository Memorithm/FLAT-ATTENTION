# M10 — Native GQA / MQA

M10 removes the equal-head-count restriction without expanding K/V.

## Logical contract

```text
Q   [batch, q_heads,  seq_len, head_dim]
K/V [batch, kv_heads, seq_len, head_dim]
O   [batch, q_heads,  seq_len, head_dim]
LSE [batch, q_heads,  seq_len]
```

The relationship is valid only when:

```text
q_heads % kv_heads == 0
```

and the group mapping is deterministic:

```text
group_size = q_heads / kv_heads
kv_head(q_head) = q_head / group_size
```

Therefore:

- MHA: `q_heads == kv_heads`, group size 1;
- GQA: `1 < kv_heads < q_heads`;
- MQA: `kv_heads == 1`.

A non-divisible relationship is rejected before dispatch.

## No physical K/V expansion

The production oracle and the M10 WGPU executor index K/V at their physical KV-head count. They never build an expanded tensor with `q_heads` copies.

For f32 storage the physical input byte counts are:

```text
Q bytes   = batch * q_heads  * seq_len * head_dim * 4
K bytes   = batch * kv_heads * seq_len * head_dim * 4
V bytes   = batch * kv_heads * seq_len * head_dim * 4
```

For example, `q_heads=16, kv_heads=1` stores each of K and V at 1/16 of the scalar count that an expanded MHA representation would require.

The permanent resident-WGPU test checks the actual buffer element metadata immediately after upload and before dispatch. No helper creates an expanded K/V allocation in the production path.

## Numerical semantics

M10 keeps the M9 stable online-softmax semantics:

1. FP32 Q·K accumulation;
2. running maximum update;
3. rescaling by `exp(old_max - new_max)`;
4. FP32 probability numerator and output accumulation;
5. FP32 LSE.

The first grouped GPU kernel deliberately uses the qualified fixed-tree Q4 portable reduction. Subgroup, vec4 and double-buffer variants are not silently inherited until each grouped variant has its own parity and benchmark evidence.

## Resident execution

`WgpuGroupedAttention` exposes:

- host convenience `forward`;
- explicit `upload`;
- `forward_resident` with no implicit readback;
- `download_attention` only when the caller requests host values.

The resident output remains packed as `[O | LSE]`, matching the existing fused-output convention.

## FLAT-specific next optimization: head-group KV reuse

Ordinary GQA avoids *storage* duplication but a naïve GPU mapping can still reload the same physical K/V tile separately for every query head that belongs to one KV head.

FLAT-ATTENTION will treat this as a first-class tiling dimension:

```text
one physical KV tile
        │
        ├── Q head h0, query rows q..q+R
        ├── Q head h1, query rows q..q+R
        └── ... heads in the same KV group
```

The intended candidate maps multiple query heads from one group into one workgroup so K/V are staged once and reused across both query rows **and query heads**. This is distinct from merely representing GQA correctly.

Promotion rules remain strict:

- no default selection before device parity;
- no claimed bandwidth reduction beyond the static load model until the generated shader is inspected and measured;
- no speedup claim before a reproducible physical-GPU benchmark;
- portable one-head-per-workgroup grouped fallback is always retained.

## SciRust integration direction

SciRust already has resident WGPU transformer execution paths. The integration target is a buffer-level adapter, not a host-side tensor conversion layer. The desired hand-off is:

```text
SciRust resident Q/K/V
        ↓
FLAT grouped fused dispatch
        ↓
SciRust resident O
```

The integration must reuse the owning device/queue and must not introduce CUDA C++, WMMA, vendor SDKs, or a project-authored C ABI bridge.
