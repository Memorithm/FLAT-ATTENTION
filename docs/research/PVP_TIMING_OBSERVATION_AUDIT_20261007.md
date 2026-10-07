# Prepared timing observation audit — 7 October 2026

PR #347 merged at `3db61bce86931b095fee69191aab8c95e6b5bf55` after all ten latest exact-head checks and all eight workflow runs succeeded on `66d5a5d45ad0df85fed0b0eb26a3b0d8af83c64f`.

M53's first attempt failed in its physical-environment step before any measurement; the retained log does not establish which suppressed probe failed or its cause. Its second attempt explicitly reported `deferred_gpu_busy`: no M53 benchmark ran and no performance evidence was accepted. A green infrastructure check is not hardware performance qualification. Native Vulkan enumeration succeeded during the read-only follow-up.

## Scope of the observation gaps

A read-only analysis of the existing PR #347 native archive found 106 gap entries across all occupancy snapshots. All were `fd_directory`, with unverified process identity and null cgroup. They concerned 106 distinct PIDs.

The retained kernel lifecycle has 62,399 events and no repeated child PID in its fork events. Reconstructing controller descendants by starting with the recorded controller PID and propagating ownership through recorded forks yields:

| Gap classification | Count |
|---|---:|
| Not a recorded controller descendant; fork and exit recorded | 100 |
| Not a recorded controller descendant; only exit recorded | 6 |

This is a reconstruction of process provenance in the visible trace. It does not identify prior device descriptors or GPU activity. A disappeared foreign process may or may not have held a target device. These gaps remain unknown; no historical sample is upgraded to admitted performance evidence. No inference about the GPU driver's correctness follows from this analysis.

The campaign still has 10,800 measured observations and 60,750 exact comparisons. Its forward packed7/vec4 paired median ratios of 1.329–1.680 remain descriptive resident wall diagnostics.

## Reproduction inputs

Use the [immutable original evidence](https://github.com/Memorithm/FLAT-ATTENTION/tree/66d5a5d45ad0df85fed0b0eb26a3b0d8af83c64f/docs/research/evidence/pvp-timing-thor-20261007). The archive SHA256 is `b8733b8cf3f49e6863acad759a64d15f8c1d5faa9963c44b7db44548213a7a93`.

Within extracted `campaign.mFjggf9M/block-1`:

| File | SHA256 |
|---|---|
| occupancy.jsonl | bc104b79bdcfc71f1d79c6da9dbf30bafaa5ba1dc244a63d29a3721aaaf50fc8 |
| kernel-lifecycle.json | 919d4c0a90005d50004934e0d118f57a0e054a69367fdf702bd668c6f3662066 |
| control.log | 995227a6754e6b19c358935834e6648097999cdb9d8e578dbd3c82a663ead48d |

Method: sum gap counts and group by stage/context; collect fork child PIDs and exit PIDs; propagate descendants from the controller PID in chronological event order; classify each gap PID against those sets. This analysis covers all snapshots, not only timed intervals, and does not replace the capture-time admission decisions.

## Next observation work

RemoteOps remains owner of the device-observation contract. A future bounded experiment should distinguish foreign process disappearance from controller-owned churn, preserve unknown identities and retain lifecycle loss/namespace boundaries. Avoid launching extra out-of-scope monitoring processes during a new timing interval where possible. Whether those monitors caused any particular historical gap is not established.

Do not relax the current gate to improve admission counts, kill unrelated processes, or silently change the observer during a cohort. Any new observation/control mechanism must be separately versioned and qualified before a new confirmatory timing campaign. The final SML model and its future harness remain independent of optional FLAT/SciRust/NNIS partners.

## Strict raw-record integrity follow-up

A separate read-only check of all three original logs found exactly 60,750 unique comparison keys and 13,500 unique forward/inverse sample keys including warmups. Their key sets exactly cover the frozen process × shape × bank × arm × trial × phase × comparison products. There are 10,800 measured samples, no duplicates or missing keys, and no mismatches.

For every check, the reported physical word count matches the layout plus four guard words: packed `G * ceil(K/128) * 4 + 4`, other arms `K * ceil(G/128) * 4 + 4`. Every wrong-word count is zero and every mismatch tuple list is empty. Each sample has positive wall/upload times, exact=true, admission=none, the expected measured/warmup label, trial/round/repeat relation and cyclic/mirrored arm position. Each process has exactly one complete marker with samples=3600, rejected_geometries=0 and failed_comparisons=0.

This checks the archived records against the protocol; it does not rerun the mathematical oracle, establish hardware inactivity or change the admission verdict.

## Future adapter admission guard

The original frozen timing source rejected CPU adapters but did not explicitly reject Other or VirtualGpu adapter identities. The follow-up requires IntegratedGpu or DiscreteGpu and tests both admitted types and all three rejected types. Historical Thor evidence used IntegratedGpu and is unaffected. This guard does not establish occupancy/exclusivity or change the old source revision; any new execution must identify its new source and binary.
