# MAA-13a Q/K-derived Boolean signature exploratory observation

Status: exploratory observation preserved. No confirmatory or performance claim.

## Provenance

- branch: `research/maa-qk-signature`
- PR: `#264`
- observed head: `9ef4c2cb838b945b81c2bc11af70d6d71920e2c8`
- preregistration commit: `18fe327704045fb44ef85f5af10ad400056820ba`
- workflow: `maa-host`
- workflow run: `35651831245`
- job: `host-oracle` (`106505936489`)
- protocol: `maa-qk-signature-exploratory/v1`
- cases: 128
- candidates per case: 16
- input dimension: 16
- signature widths: 8, 16, 32, 64, 128

The workflow executed the Q/K signature example twice and required byte-for-byte
identical CSV output. Formatting, strict Clippy, debug/release MAA tests,
MAA-10/11/12 smokes and strict documentation also passed on the observed head.

## Aggregate observation

```csv
row_type,width,threshold,top1_matches,top2_rank_hits,pair_agree,pair_disagree,pair_tied,selected,admission_top2_hits,false_negatives,empty_cases,scale_signature_collisions,scale_strict_exact_orderings,scale_unresolved_hamming_ties
ranking,8,NA,30,93,8568,4042,2750,NA,NA,NA,NA,NA,NA,NA
admission,8,2,30,93,8568,4042,2750,329,112,144,5,NA,NA,NA
admission,8,3,30,93,8568,4042,2750,766,179,77,0,NA,NA,NA
admission,8,4,30,93,8568,4042,2750,1236,231,25,0,NA,NA,NA
scale_control,8,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,64,64,64
ranking,16,NA,44,119,9823,3785,1752,NA,NA,NA,NA,NA,NA,NA
admission,16,4,44,119,9823,3785,1752,145,74,182,41,NA,NA,NA
admission,16,6,44,119,9823,3785,1752,541,175,81,1,NA,NA,NA
admission,16,8,44,119,9823,3785,1752,1163,232,24,0,NA,NA,NA
scale_control,16,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,64,64,64
ranking,32,NA,58,131,10851,3452,1057,NA,NA,NA,NA,NA,NA,NA
admission,32,8,58,131,10851,3452,1057,56,42,214,81,NA,NA,NA
admission,32,12,58,131,10851,3452,1057,387,163,93,5,NA,NA,NA
admission,32,16,58,131,10851,3452,1057,1112,243,13,0,NA,NA,NA
scale_control,32,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,64,64,64
ranking,64,NA,70,163,11926,2795,639,NA,NA,NA,NA,NA,NA,NA
admission,64,16,70,163,11926,2795,639,18,17,239,112,NA,NA,NA
admission,64,24,70,163,11926,2795,639,282,163,93,11,NA,NA,NA
admission,64,32,70,163,11926,2795,639,1065,245,11,0,NA,NA,NA
scale_control,64,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,64,64,64
ranking,128,NA,87,175,12880,2119,361,NA,NA,NA,NA,NA,NA,NA
admission,128,32,87,175,12880,2119,361,5,5,251,123,NA,NA,NA
admission,128,48,87,175,12880,2119,361,209,148,108,22,NA,NA,NA
admission,128,64,87,175,12880,2119,361,1030,251,5,0,NA,NA,NA
scale_control,128,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,64,64,64
```

## Interpretation

### Angular information is materially present

On the frozen unit-normalized panel, increasing signature width improved every
recorded ranking diagnostic in the observed sequence:

- top-1 exact matches: 30, 44, 58, 70, 87 for widths 8..128;
- dense-top2 rank intersections: 93, 119, 131, 163, 175;
- strict pairwise agreements increased from 8,568 to 12,880;
- Hamming ties decreased from 2,750 to 361.

This is exploratory evidence that the Q/K-derived projection carries useful
angular ordering information on this synthetic panel. It is not a model-quality
or generalization claim.

### The half-width admission arm is the strongest observed logical frontier

For the preregistered `B/2` M13B.2 Hamming threshold:

- width 8 selected 1,236 candidate evaluations and retained 231/256 dense-top2
  occurrences;
- width 16 selected 1,163 and retained 232/256;
- width 32 selected 1,112 and retained 243/256;
- width 64 selected 1,065 and retained 245/256;
- width 128 selected 1,030 and retained 251/256.

All five `B/2` arms had zero empty selections. The 128-bit arm therefore
provides a concrete exploratory reason to continue studying Q/K-derived Boolean
routing: it removed 1,018 of the 2,048 logical candidate score evaluations in
this host panel while missing 5 of 256 dense-top2 occurrences.

These are logical counts only. They do not establish physical work avoided,
latency, bandwidth, energy, memory benefit or GPU usefulness.

### Positive-scale invariance is a hard information boundary

For every tested width, all 64 low/high positive-scale pairs produced identical
signatures. All 64 pairs had a strict exact-score ordering, and Hamming left all
64 unresolved.

This is not an implementation defect: it is the preregistered negative control
for a sign-only homogeneous projection. It means that F2 or Zhegalkin functions
computed solely from these scale-invariant signature bits cannot recover the
missing positive-magnitude information by themselves.

## Research consequence

MAA-13a justifies a new exploratory slice, but not a confirmatory promotion.

The next hypothesis should preserve the successful angular signal while adding
an independent norm-aware key feature. Since, for one fixed query, exact
`q.k = ||q|| * ||k|| * cos(theta)` has a common query-norm factor across
candidate keys, a compact key-norm side channel is the relevant missing
candidate-dependent magnitude term.

A successor MAA-13b should therefore test a frozen norm-aware Boolean/F2/
Zhegalkin policy on a fresh variable-norm exploratory generator. It must include
the MAA-13a Hamming-only path as a baseline and must not reuse the MAA-11b
confirmatory holdout.

## Non-claims

This result does not establish real-model quality, latency, throughput,
bandwidth, energy efficiency, physical memory savings, novelty, GPU usefulness
or production readiness. It does not replace exact q.k, softmax, dense FLAT or
the existing fallback contract.
