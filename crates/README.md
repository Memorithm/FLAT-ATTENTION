# Workspace crates

This directory holds supporting packages for FLAT-ATTENTION. The reusable public
contract lives in the root `flat-attention` crate (`src/`, `flat_attention::api::v1`).
Nothing in `crates/` is a silent replacement for that contract.

## Production-adjacent / reusable building blocks

| Crate | Role |
|-------|------|
| `epg-core` | Runtime-neutral Elastic Positional Geometry types. No attention, cache, or GPU code. |
| `flat-epg-reference` | Deterministic CPU oracle fusing hybrid SO(2)/SO(4) rotations into grouped attention. |
| `flat-epg-wgpu` | Correctness-first WGPU qualification path for EPG. |
| `flat-elastic-kernel` | Host-side Elastic/FLAT planning surface used by SciRust opt-in features. |

## Qualification / candidate crates

These packages exist to qualify a specific generation against the oracle. They
are **not** default routing and carry no standalone performance claim.

| Crate | Role |
|-------|------|
| `flat-ada-a1-candidate` | ADA A1 shader candidate plus device evidence. |
| `flat-ada-graduation` | Graduation helpers for promoting a qualified candidate. |
| `flat-epg-q4-candidate` | Opt-in Q4 tiling with EPG fused into K/V staging. |
| `flat-da-luc-tier-qualification` | Research DA-LUC tier evidence / qualification. |

## Research semantic stack

The `flat-semantic*` crates describe experimental structured-history / semantic
control surfaces. They remain outside `api::v1` until an explicit promotion gate
says otherwise. See `docs/research/` and `docs/SEMANTIC_*`.

## How to treat these crates

- Do not import a candidate crate from SciRust production code unless a
  documented feature flag and qualification record exist.
- Do not interpret a crate name as a public API promise.
- New experimental work should land here (or under `docs/research/`) rather
  than expanding `api::v1` by default.
