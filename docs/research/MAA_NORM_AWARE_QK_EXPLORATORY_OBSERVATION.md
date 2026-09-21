# MAA-13b norm-aware Q/K exploratory observation

Status: exploratory observation preserved. No confirmatory or performance claim.

## Provenance

- branch: `research/maa-norm-aware-qk`
- PR: `#265`
- observed head: `27954581528f5531649e418097fd0c37cdd58ae2`
- preregistration commit: `0376b668fafeefddd207c8821451fcd7686c0345`
- workflow: `maa-host`
- workflow run: `35653103021`
- job: `host-oracle` (`106510007708`)
- protocol: `maa-norm-aware-qk-exploratory/v1`
- cases: 128
- candidates per case: 16
- Q/K dimension: 16
- signature width: 128 bits

The workflow executed the MAA-13b sweep twice and required byte-for-byte
identical CSV output. Formatting, strict Clippy, debug/release tests and all
earlier MAA smokes also passed on the observed head.

## Aggregate observation

```csv
arm,selected,top2_hits,false_negatives,empty_cases,output_error_sum,lse_error_sum,comparable_cases,additional_vs_near,additional_top2_hits
dense,2048,256,0,0,0.000000000,0.000000000,128,0,0
hamming_near,1036,234,22,0,30.313804522,58.125929475,128,0,0
hamming_medium,1531,255,1,0,13.757663224,20.800710917,128,0,0
norm_only,1028,214,42,0,39.791734681,84.552327752,128,0,0
norm_aware_maa,1299,255,1,0,21.835016251,37.167891860,128,263,21
matched_random,1299,155,101,0,33.039820321,62.055102468,128,0,0
```

## Interpretation

### The norm side channel repairs most strict-Hamming relevance loss

Relative to `hamming_near`, the frozen norm-aware rule:

- selected 263 additional candidates;
- recovered 21 additional dense-top2 occurrences;
- reduced false negatives from 22 to 1;
- reduced aggregate output-error sum from 30.313804522 to 21.835016251;
- reduced aggregate LSE-error sum from 58.125929475 to 37.167891860.

The additional-top2 count exactly matches the recovered top2 count on this
panel. This is exploratory evidence that the independent key-norm predicate
contains candidate-relevance information absent from the sign-only signature.

### Norm-aware MAA is more selective than simply relaxing Hamming

`norm_aware_maa` and `hamming_medium` both retained 255/256 dense-top2
occurrences with one false negative and no empty cases.

However:

- `norm_aware_maa` selected 1,299 candidates;
- `hamming_medium` selected 1,531 candidates.

The norm-aware rule therefore rejected 232 candidates that the relaxed Hamming
envelope would have retained while preserving the same top2 count on this
panel.

This is a logical candidate-count observation, not a physical speedup claim.

### Top-2 retention is not sufficient as a quality proxy

Despite identical top2 retention, `hamming_medium` had lower numerical error:

- output-error sum: 13.757663224 versus 21.835016251;
- LSE-error sum: 20.800710917 versus 37.167891860.

Therefore candidates outside the dense top-2 still contribute materially to the
softmax normalization/output on this panel. Future MAA routing evidence should
not optimize or promote a policy from top-k retention alone.

### The structured rule strongly exceeds density-matched random selection

At the same aggregate selected count (1,299):

- norm-aware MAA retained 255 dense-top2 occurrences;
- matched random retained 155;
- norm-aware output/LSE error sums were both lower than matched random.

This is bounded evidence that the frozen Q/K+norm structure is informative on
this synthetic generator; it is not evidence of real-model generalization.

### Norm alone is insufficient

The `norm_only` arm retained only 214/256 dense-top2 occurrences and had the
largest output/LSE error sums among the structured arms. Magnitude is useful as
a side channel, not as a replacement for angular similarity.

## Research consequence

MAA-13b justifies further exploration, but not confirmatory promotion.

The key methodological finding is that top2 recall can conceal meaningful
softmax-output degradation. A successor slice should first add dense-reference
**retained softmax probability mass** (or an equivalent exact mass diagnostic)
to the evidence model. Only then should a finer tiered norm/angular policy be
judged.

A plausible later policy family is a nested angular/norm ladder, for example
retaining all very-near candidates while using progressively stronger norm
requirements for wider Hamming bands. Such a rule must be preregistered on a
fresh exploratory dataset and compared with relaxed-Hamming and density-matched
controls under both retained-mass and O/LSE diagnostics.

## Non-claims

This result does not establish real-model quality, novelty, latency, throughput,
bandwidth, energy efficiency, physical memory savings, GPU usefulness or
production readiness. It does not replace exact q.k, softmax or dense FLAT.
