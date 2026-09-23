# MAA-14c coverage-7/8 confirmatory observation

Status: preserved confirmatory holdout observation under the preregistered protocol. No threshold or decision rule is changed after observation.

## Provenance

- protocol: `maa-adaptive-repair-confirmatory/v1`
- preregistration: merged PR #280
- implementation: merged PR #281
- observed exact head: `6a15673454e011d85b5c109135472638aabecb21`
- implementation merge: `d60f206fd137d94b115463819fbc064b3acbd2bd`
- workflow: `maa-host`
- run: `35857320979`
- job: `107168800948`
- holdout: 1,024 cases x 32 candidates, D=16
- release evidence executed twice with byte-identical CSV output

## Frozen-arm results

| arm | repaired rows | final selected | density | mass mean | mass p10 | mass p90 | output-error sum | LSE-error sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0 | 32,768 | 1.000000000 | 1.000000000000 | 1.000000000000 | 1.000000000000 | 0.000000000 | 0.000000000 |
| never repair | 0 | 22,990 | 0.701599121 | 0.822922282150 | 0.739846643745 | 0.896437389060 | 92.468132529 | 202.573942423 |
| coverage below 7/8 | 174 | 23,668 | 0.722290039 | 0.839181593539 | 0.767586089326 | 0.908624466638 | 85.546405971 | 181.911180019 |
| matched random 7/8 | 174 | 23,668 | 0.722290039 | 0.835714418191 | 0.765621096733 | 0.903310361512 | 87.148997072 | 186.125354290 |
| always repair | 1,024 | 24,843 | 0.758148193 | 0.866653989300 | 0.800183830215 | 0.929577069691 | 72.757961214 | 148.506730080 |
| retained-mass oracle | 935 | 24,763 | 0.755706787 | 0.865118430247 | 0.800183830215 | 0.923616837547 | 73.575784784 | 150.187459230 |

Every arm retained all 2,048 dense top-2 occurrences and had zero empty selections.

The base fell below the frozen 0.90 retained-mass diagnostic on 935 / 1,024 rows. The 7/8 trigger repaired 174 of those rows, with precision 1.0 and recall 0.186096256684 against that diagnostic label.

## Frozen confirmatory decision

The harness emitted:

`CONFIRMATORY_PASS`

with:

- additional-candidate fraction versus full repair: **0.365893146249**;
- recovered retained-mass-mean gap: **0.371796859735**.

All nine preregistered requirements pass:

1. zero empty selections: PASS;
2. no more top-2 false negatives than base: PASS (0 vs 0);
3. repaired rows strictly between zero and 1,024: PASS (174);
4. final selected count strictly between base and full repair: PASS (23,668 between 22,990 and 24,843);
5. retained-mass mean and p10 strictly above base: PASS;
6. output-error and LSE-error sums strictly below base: PASS;
7. retained-mass mean strictly above exact-cardinality matched random: PASS;
8. added-candidate fraction <=0.60: PASS (0.365893146249);
9. retained-mass-gap recovery >=0.20: PASS (0.371796859735).

## Interpretation

The fresh holdout reproduces the exploratory direction without retuning.

The unchanged structural 7/8 trigger remains a genuine intermediate point: it uses about 36.6% of the extra logical candidates required by complete hamming-medium widening while recovering about 37.2% of the retained-mass-mean gap on this holdout.

At identical final cardinality, the structural repair arm again exceeds its frozen random control on retained-mass mean and has lower aggregate O/LSE error. This supports the narrower claim that the selected hamming-medium additions carry useful structure relative to this matched random expansion.

This is still synthetic host evidence. Candidate-count reduction is not a latency, bandwidth, energy, cache-traffic, or GPU-utilization measurement.

## Decision

The unchanged `coverage_below_7_8` repair trigger is **confirmed for MAA-14d physical qualification**.

MAA-14d may now preregister a portable physical experiment that measures:

- router/trigger construction cost;
- first sparse-pass latency;
- repair-decision overhead;
- second sparse-pass/final-result latency;
- candidate/index transfer and memory traffic where measurable;
- numerical O/LSE parity against the host oracle;
- base, adaptive 7/8, matched-random, full-envelope and dense controls on the same device.

This observation does not authorize default production routing or any physical speedup claim.
