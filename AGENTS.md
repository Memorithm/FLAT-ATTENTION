# FLAT-ATTENTION Agent Bootstrap Contract

Before any autonomous coding, kernel promotion, performance claim, cross-repository integration, architectural decision, PR creation, or merge decision, read both:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/FLAT_ATTENTION_ECOSYSTEM_ROADMAP.yaml && \
git show origin/main:ROADMAP.md
```

`ROADMAP.md` is the published engineering milestone plan. The off-main agent roadmap is authoritative for persistent ecosystem boundaries, integration contracts, and promotion state.

For any machine-learning attention, dtype, backward/training, KV, runtime-adapter, benchmark, or cross-repository ML work, also read:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/ML_MATURITY_5_OF_5.yaml
```

The ML maturity overlay makes 5/5 an evidence-backed exit criterion. Never promote logical I/O counts into physical bandwidth claims, lavapipe into real-GPU performance evidence, or an inference-only slice into complete training support.

## Mandatory DA-LUC research overlay

For any compressed/quantized KV view, codebook/PQ/LUT scoring, sparse outlier residual, direct compressed decode, dynamic KV precision tiering, NNIS DA-LUC adapter, or PHR-Lite routing work, also read:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/DA_LUC_RESEARCH_PROGRAM.yaml
```

DA-LUC is a research program, not a pre-established novelty or performance result. The attention-facing property to prove is direct consumption of an explicit versioned compressed representation without dense K/V materialization when the selected backend supports it. Do not call this "zero dequantization" when scalar/register conversion still occurs.

Any KV compression claim must use exact effective bits/value including codebooks, scales/zero-points, residual values and indices/bitmaps, page metadata, alignment, and padding. A nominal 8x-16x target is never evidence. K and V are explicitly asymmetric research axes and must not be forced into the same representation without measurement.

PHR-Lite is restricted to optional KV precision/retention/page/offload/eviction routing after deterministic recency and attention-mass baselines exist. It does not replace tokenization, next-token prediction, the LM head, or the language-model objective.

If any required roadmap or applicable overlay cannot be read, fail closed for major architecture, performance-promotion, representation-contract, compressed-KV, cross-repository contract, or merge decisions. Read-only diagnosis is allowed.

## Non-negotiable rules

- correctness hierarchy: mathematical definition → deterministic scalar Rust oracle → portable GPU path → optimized GPU path;
- never disguise CPU fallback as GPU execution;
- no performance claim without reproducible real-device evidence;
- no required project-authored CUDA C/C++/vendor SDK layer in the core architecture;
- no compressed-KV promotion without a dense scalar/reference KV oracle;
- compressed-KV schemas must be versioned and fail closed on unknown or malformed metadata;
- exact compressed storage accounting must include every overhead, not nominal code width only;
- quality, memory, latency, and tokens/s must be reported together before real-model compressed-KV promotion;
- required CI must be green on the exact PR head before merge;
- cross-repository contracts with SciRust, ElasticXxx, SLHAv2, NNIS, Forge, Verify, or Hub are never assumed from similar code alone;
- 5/5 maturity cannot be claimed until the applicable end-to-end, interoperability, real-hardware, numerical and evidence gates in the ML overlay are closed.

Reread the roadmaps and applicable ML/DA-LUC overlays at session start, before selecting a major milestone, after benchmark promotion/rejection or strategy changes, before cross-repository work, and before relevant PR/merge decisions.

Do not merge the off-main roadmap or research overlays themselves into `main` unless the user explicitly requests it.
