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

Treat root `AGENTS.md` as mandatory bootstrap policy. If either roadmap or the applicable ML overlay is unavailable, fail closed for major architecture, performance-promotion, ecosystem-contract, or merge decisions.

Preserve oracle-first correctness, real-device benchmark honesty, exact-head green CI before merge, repository ownership boundaries, and the rule that `5/5` is earned only by the overlay's end-to-end evidence gates.
