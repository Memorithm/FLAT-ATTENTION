# MAA-11b Frozen Confirmatory Holdout — First CI Observation

Status: **confirmatory result preserved; no retuning permitted**

This document preserves the first successful CI observation of the frozen MAA-11b synthetic confirmatory holdout. It records the result produced by the already-frozen policy and dataset protocol; it does not modify the freeze, policy, generator, metrics, or decision rule.

## Provenance

- PR: `#241` (`research/maa-confirmatory-holdout`)
- harness head observed by CI: `57fd5cc32f7857f77a5d831bc909a4ced8bb08f3`
- frozen policy source revision: `4030b229a2ff329c1263cad677812d659422b448`
- workflow: `maa-host`
- workflow run: `35014854280`
- job: `host-oracle` (`104535599346`)
- runner: GitHub-hosted `ubuntu-24.04`, image `20260907.300.1`, x86_64
- Rust: `rustc 1.89.0 (29483883e 2025-08-04)`
- protocol: `maa-confirmatory-holdout/v1`
- cases: 16 deterministic cases
- candidates per case: 8
- dense relevance target: frozen top-2 unscaled `q·k` with lower candidate ID tie-break

The workflow executed the confirmatory example twice and required byte-for-byte identical CSV output before reporting success.

## Frozen aggregate observation

The first successful CI run emitted the following aggregate row verbatim:

```csv
aggregate,ALL,-,-,-,76,9,15,2,17,30,2.076023623,5.068914294,4.964793444,14.441770077,0,9,7,67,FRONTIER_REJECT
```

Under the frozen CSV schema, this means:

| Metric | Boolean-only control | Frozen Boolean + F2 + Zhegalkin policy |
| --- | ---: | ---: |
| score evaluations | 76 | 9 |
| dense-top2 hits retained | 15 | 2 |
| false negatives vs dense top-2 | 17 | 30 |
| non-empty output-error sum | 2.076023623 | 5.068914294 |
| non-empty LSE-error sum | 4.964793444 | 14.441770077 |
| empty sparse selections | 0 | 9 |

Additional frozen aggregates:

- comparable non-empty cases: `7`
- additional score evaluations avoided relative to Boolean-only control: `67`
- frozen classification: **`FRONTIER_REJECT`**

## Interpretation

The confirmatory result is negative for the frozen MAA-11b frontier rule. Although the multi-algebra policy issued fewer score evaluations on this synthetic panel, it failed the preregistered quality/frontier conditions: retained fewer dense-top2 targets, produced more false negatives, accumulated larger output and LSE error over comparable non-empty cases, and produced nine empty selections where the Boolean-only control produced none.

This result must be retained as negative evidence. It must not be converted into a pass by changing thresholds, policy structure, feature schema, dataset identity, generator, metrics, or decision rule after observing the outcome.

## Scientific boundary

This is a small deterministic synthetic confirmatory holdout. It does **not** establish:

- real-model quality;
- GPU or CPU speedup;
- latency, throughput, bandwidth, energy, or memory improvement;
- production readiness;
- novelty of the Boolean/F2/Zhegalkin construction;
- impossibility of other multi-algebra policies.

The result applies only to the frozen MAA-11b policy/protocol identified above. Any successor hypothesis must be a new preregistered series with a new tuning/confirmatory separation; this holdout must not be reused for tuning that successor.
