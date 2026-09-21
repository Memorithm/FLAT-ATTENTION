# FA-V888-1 — Structural Candidate Routing Host Oracle

Status: research-only host correctness slice. No runtime routing or performance claim.

## Purpose

This slice introduces the first executable structure needed by the BANC v888 programme without embedding BANC data in FLAT-ATTENTION.

The research object is a canonical per-query candidate list:

```text
query row -> sorted unique key positions
```

stored in a CSR-like `offsets + key_positions` form.

## Contract

The representation:

- is tied to one validated FLAT self-attention shape;
- stores one row per `[batch, head, query]`;
- sorts key positions deterministically;
- rejects duplicate keys;
- rejects out-of-range keys;
- permits an empty metadata row so routing failures can be represented;
- exposes explicit causal intersection;
- reports exact admitted-pair counts.

Numerical attention does **not** fabricate an output for an empty effective survivor set. It fails closed.

## Numerical qualification

Two scalar host paths are retained:

1. dense masked reference: scans every key and checks structural membership;
2. sparse structural reference: visits only canonical candidate keys.

For one identical candidate set, both paths must be bit-identical because retained keys are evaluated in the same ascending order.

Required controls include:

- arbitrary sparse candidate rows;
- causal intersection;
- malformed geometry;
- duplicate/out-of-range keys;
- empty effective survivor;
- all-accept non-causal parity against the existing `forward_reference`;
- all-accept causal parity against the existing `forward_reference`.

## Boundaries

FA-V888-1 does not:

- load or encode BANC v888;
- define how graph nodes map to language tokens;
- use MAA/F2/Zhegalkin predicates yet;
- execute WGPU;
- claim fewer physical K/V reads;
- claim latency, quality, energy, bandwidth, or memory improvements;
- alter `api::v1` or default routing.

The next step, FA-V888-2, is the matched structural-mask family. MAA structural cooperation begins only in FA-V888-3 after the host candidate semantics are qualified.
