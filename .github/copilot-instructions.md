# FLAT-ATTENTION repository agent instructions

Before repository changes, read the persistent ecosystem roadmap and the published engineering roadmap:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/FLAT_ATTENTION_ECOSYSTEM_ROADMAP.yaml && \
git show origin/main:ROADMAP.md
```

For attention, dtype, backward/training, KV, runtime-adapter, benchmark, or cross-repository ML work, also read:

```bash
git show origin/agent/ecosystem-roadmap:.agent/ML_MATURITY_5_OF_5.yaml
```

For compressed/quantized KV views, codebook/PQ/LUT scoring, sparse outlier residuals, direct compressed decode, dynamic KV precision tiers, NNIS DA-LUC adapters, or PHR-Lite KV routing, also read:

```bash
git show origin/agent/ecosystem-roadmap:.agent/DA_LUC_RESEARCH_PROGRAM.yaml
```

Treat root `AGENTS.md` as mandatory bootstrap policy. If either roadmap or any applicable overlay is unavailable, fail closed for major architecture, performance-promotion, representation-contract, compressed-KV, ecosystem-contract, or merge decisions.

DA-LUC remains research-only until its oracle, exact-storage, quality and real-hardware gates pass. Do not promote nominal index width into a memory claim, call a path "zero dequantization" when conversion remains, or report an 8x-16x target without measured effective bits/value including codebooks, residuals, metadata, alignment and padding. Preserve K/V asymmetry as an explicit measured design choice.

PHR-Lite is limited to optional KV routing and must beat simpler recency/attention-mass controls under the same storage and quality budget before promotion. It does not replace tokenization or the language-model objective.

Preserve oracle-first correctness, real-device benchmark honesty, exact-head green CI before merge, repository ownership boundaries, and the rule that `5/5` is earned only by the overlay's end-to-end evidence gates.
