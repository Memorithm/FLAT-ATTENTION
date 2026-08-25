# M60 — Q1 direct K/V load candidate

M60 continues the Phase O optimization loop after the physical-Thor M59 result. M59 proved that the M58 `Q1Vec4Mha` execution shape is a large improvement over the older Q4 MHA route, but it still loses to SciRust's resident multi-dispatch baseline in three long-context rows: `seq_len=512, D64` causal and non-causal, and `seq_len=512, D128` non-causal.

## Accepted baseline

The M59 product-side benchmark used the physical `NVIDIA Tegra NVIDIA Thor` Vulkan adapter, batch 1, one MHA head, `seq_len` 128/512, D64/D128, causal/non-causal, with correctness gating before timing and upload/readback excluded.

M58 Q1 beat the contemporaneous Q4 baseline in all eight measured rows, but the remaining SciRust-naive gaps at `seq_len=512` were material:

- D64 non-causal: SciRust naive about 773 us vs M58 Q1 about 1987 us;
- D64 causal: SciRust naive about 782 us vs M58 Q1 about 1024 us;
- D128 non-causal: SciRust naive about 1187 us vs M58 Q1 about 2154 us.

These measurements identify long-context work inside each Q1 workgroup as the next bottleneck rather than dispatch count alone.

## Bottleneck hypothesis

M58 kept the Q4 K/V staging design even though its execution geometry changed fundamentally.

In Q4, several query rows share one staged K/V tile, so workgroup-memory staging amortizes global loads. In Q1, one workgroup computes exactly one query row. Every K/V vec4 loaded into `k_shared` / `v_shared` is consumed exactly once by that workgroup before the next key/tile. The staging therefore adds:

- workgroup-memory stores for K and V;
- workgroup-memory reloads for K and V;
- one tile synchronization boundary;
- address arithmetic for tile-to-shared remapping;

without any cross-query reuse.

## Candidate

The isolated `flat-m60-q1-direct-candidate` crate changes only that mechanism:

- Q remains register-resident exactly as in M58;
- the 64-lane dot-product reduction is unchanged;
- the online-softmax shared state and synchronization are unchanged;
- each active lane loads its K and V `vec4<f32>` directly from storage for the current key row;
- `k_shared`, `v_shared`, KV-tile staging and the staging barrier are removed.

The candidate remains MHA-only and D64/D128-only. It rejects GQA/MQA and other head dimensions before dispatch.

## Qualification gates

Before any retention or routing decision:

1. host contract tests must reject GQA and non-D64/D128 shapes;
2. the shader source must contain no K/V workgroup staging arrays;
3. WGPU output and LSE must match `forward_reference_grouped` for D64/D128, causal and non-causal;
4. the full workspace CI must stay green;
5. physical Thor must compare M60 against the accepted M58 Q1 path under the established GPU lock, idle/cooldown, contamination-watch and alternating-order protocol;
6. the SciRust naive baseline remains the product-level performance gate.

## Decision boundary

This PR is a performance-candidate implementation only. It changes no production router and makes no generic speedup claim.

If direct K/V loads do not materially improve the long-context rows, M60 is rejected and the next isolated mechanism will target M58's per-key 64-lane workgroup reduction/synchronization, most likely through capability-gated subgroup collectives with a portable fallback.

`performance_claim=none`
