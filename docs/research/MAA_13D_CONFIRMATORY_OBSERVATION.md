# MAA-13d tiered-recall confirmatory observation

Status: confirmatory holdout observed under the preregistered protocol merged in PR #271. This record preserves the result and applies the frozen decision rule without retuning.

## Provenance

- implementation PR: #272
- qualified head: `d64e50b96bf380b46d502a090f7cbd144383daae`
- merge commit: `44fa5ee6db80c90c97278ec0885f91c26abd2aef`
- workflow: `maa-host`
- run: `35818830203`
- panel: 512 cases x 32 candidates, D=16
- protocol: `maa-tiered-recall-confirmatory/v1`

The exact-head CI, MAA host suite, semver, CodeQL and supply-chain checks passed. The holdout CSV was generated twice and required byte-for-byte equality.

## Frozen candidate result

`tiered_recall`:

- selected: 11,534 / 16,384
- density: 0.703979492
- dense top-2 hits: 1,024 / 1,024
- false negatives: 0
- retained-mass mean: 0.822473061597
- retained-mass minimum: 0.517729937745
- p10: 0.735914907053
- p50: 0.826574208751
- p90: 0.904700413066
- cases >=0.90: 60 / 512
- cases >=0.95: 3 / 512
- empty cases: 0

Frozen comparators:

- `hamming_medium`: density 0.760925293, top-2 false negatives 0, retained-mass mean 0.867201619890, p10 0.795034780837.
- MAA-13c `norm_aware`: density 0.640014648, top-2 false negatives 0, retained-mass mean 0.770179070134, p10 0.674772532927.
- matched random at the MAA-13c comparator density: retained-mass mean 0.639305451525, p10 0.496793386401.

## Frozen decision rule

The preregistered requirements are all satisfied:

1. zero empty selections: PASS (0);
2. no more top-2 false negatives than hamming-medium: PASS (0 versus 0);
3. lower density than hamming-medium: PASS (0.703979492 < 0.760925293);
4. retained-mass mean and p10 above frozen MAA-13c: PASS (0.822473061597 > 0.770179070134; 0.735914907053 > 0.674772532927);
5. retained-mass mean and p10 above the recorded matched-random control: PASS.

Decision: `CONFIRMATORY_PASS`.

## Interpretation boundary

This confirms the preregistered synthetic direction: the frozen tiered-recall policy trades additional density relative to MAA-13c for materially higher retained softmax mass while remaining sparser than hamming-medium, without increasing top-2 false negatives on this holdout.

It does not establish real-model quality, novelty, physical work avoided, latency, throughput, bandwidth, energy efficiency, GPU usefulness or production readiness. The preregistration permits proceeding to MAA-13e physical qualification; it does not authorize default routing.
