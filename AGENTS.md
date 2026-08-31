# FLAT-ATTENTION Agent Bootstrap Contract

Before any autonomous coding, kernel promotion, performance claim, cross-repository integration, architectural decision, PR creation, or merge decision, read both:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/FLAT_ATTENTION_ECOSYSTEM_ROADMAP.yaml && \
git show origin/main:ROADMAP.md
```

`ROADMAP.md` is the published engineering milestone plan. The off-main agent roadmap is authoritative for persistent ecosystem boundaries, integration contracts, and promotion state.

If either required roadmap cannot be read, fail closed for major architecture, performance-promotion, cross-repository contract, or merge decisions. Read-only diagnosis is allowed.

## Non-negotiable rules

- correctness hierarchy: mathematical definition → deterministic scalar Rust oracle → portable GPU path → optimized GPU path;
- never disguise CPU fallback as GPU execution;
- no performance claim without reproducible real-device evidence;
- no required project-authored CUDA C/C++/vendor SDK layer in the core architecture;
- required CI must be green on the exact PR head before merge;
- cross-repository contracts with SciRust, ElasticXxx, SLHAv2, NNIS, Forge, Verify, or Hub are never assumed from similar code alone.

Reread both roadmaps at session start, before selecting a major milestone, after benchmark promotion/rejection or strategy changes, before cross-repository work, and before relevant PR/merge decisions.

Do not merge the off-main roadmap itself into `main` unless the user explicitly requests it.
