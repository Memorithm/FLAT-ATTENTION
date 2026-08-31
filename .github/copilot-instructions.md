# FLAT-ATTENTION repository agent instructions

Before repository changes, read the persistent ecosystem roadmap and the published engineering roadmap:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/FLAT_ATTENTION_ECOSYSTEM_ROADMAP.yaml && \
git show origin/main:ROADMAP.md
```

Treat root `AGENTS.md` as mandatory bootstrap policy. If either roadmap is unavailable, fail closed for major architecture, performance-promotion, ecosystem-contract, or merge decisions.

Preserve oracle-first correctness, real-device benchmark honesty, exact-head green CI before merge, and repository ownership boundaries.
