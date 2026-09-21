# MAA-12a survivor-floor exploratory observation

Status: exploratory observation preserved. No confirmatory claim.

## Provenance

- branch: `research/maa-survivor-floor`
- PR: `#262`
- observed head: `74bd1f5b179fa1ed76f85c78594dad700f003da1`
- preregistration commit: `e6afc9ee89a71584087b7322e892c620d1b13b8b`
- workflow: `maa-host`
- workflow run: `35631149145`
- job: `host-oracle` (`106437379569`)
- protocol: `maa-survivor-floor-exploratory/v1`
- generator: SplitMix64 seed `0x004d_4141_2d31_3261`
- cases: 64
- candidates per case: 8
- head dimension: 2

The workflow ran the exploratory CSV twice and required byte-for-byte equality.

## Aggregate observation

```csv
arm,floor,score_evaluations,top2_hits,false_negatives,empty_selections,output_error_sum,lse_error_sum,rescued_candidates
boolean,NA,246,63,65,1,16.747797560,50.673272491,0
floor,0,33,13,115,37,16.296826065,54.565392375,0
floor,1,69,22,106,1,39.376145273,131.913585663,36
floor,2,126,33,95,1,26.837378703,91.351844907,93
floor,3,179,45,83,1,19.908654690,68.412536681,146
floor,4,216,53,75,1,18.128778871,57.384174824,183
```

## Interpretation

The survivor floor behaves structurally as intended:

- floor 0 exposes the severe collapse of the strict conjunction: 37 empty
  selections across 64 cases;
- every positive tested floor reduces empty selections to 1, matching the
  Boolean-only empty count;
- increasing the floor monotonically increases selected-score evaluations and
  recovers more dense-top2 targets;
- the rescue remains bounded by Boolean admission.

However, no tested floor dominates the Boolean-only control on this exploratory
panel.

At floor 4, the closest tested arm uses 216 selected-score evaluations versus
246 for Boolean-only, but retains 53 dense-top2 hits versus 63, with 75 false
negatives versus 65. Its output-error sum (18.128778871) and LSE-error sum
(57.384174824) also remain above Boolean-only (16.747797560 and 50.673272491).

Therefore this observation does **not** justify freezing a MAA-12b confirmatory
survivor-floor policy. The result is evidence against the specific hypothesis
that generic F2/Zhegalkin support-count rescue is sufficient to recover the
Boolean-only work/quality frontier.

## Research consequence

The next MAA hypothesis should address the likely structural weakness directly:
the auxiliary predicates in MAA-11b/12a are not derived from numerical Q/K
similarity. A successor series should investigate Boolean/F2/Zhegalkin features
derived from Q/K signatures or sketches, then compare their ranking/rejection
behavior against exact dense Q.K relevance on a fresh exploratory dataset.

That successor must remain a new research series. It must not alter the
preserved MAA-11b holdout result, and any eventual confirmatory run must use a
new frozen policy identity and distinct tuning/holdout dataset identities.

## Non-claims

This synthetic host observation does not establish model quality, hardware
speedup, latency, bandwidth, energy, memory savings, novelty, or production
readiness. It does not change the stable API, default routing, or GPU kernels.
