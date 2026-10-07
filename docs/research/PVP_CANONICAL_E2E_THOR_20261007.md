# Canonical end-to-end PVP on Thor — 2026-10-07

Verdict: ten exact native processes and a repeatable descriptive advantage for the existing packed7 canonical-input/output carrier. Final gain confirmation is **not admitted**.

## Execution identity and complete coverage

Prospective protocol: [PVP_CANONICAL_E2E_PROTOCOL.md](PVP_CANONICAL_E2E_PROTOCOL.md).
Execution source: `b020d31711d75807eb0f8b13a6a967a8522c5d1c`.
Engineering PR #349 merge: `666abc4683107da2c590f5b8b5759b6557858b27`, after 15/15 latest exact-head checks and 11/11 workflows succeeded.
Native executable SHA256: `c27fdd574440886f2fe81eaf11b28aaa6ee5f6d86e9bdf6c5d6ab7c6fee69aec`.

Ten numbered fresh native processes, each in a separately acquired/released scoped reservation, completed with exit 0. There are 14,400 measured wall intervals, 3,600 warmups and 72,000 complete physical/canonical comparisons. No wrong words, duplicate/missing record keys or rejected geometries were found by the strict final verifier. Adapter identity is NVIDIA Tegra NVIDIA Thor, IntegratedGpu, Vulkan, installed driver 580.00; native compiler is Rust 1.89.0, release profile.

Both arms begin/end with identical gate-major u64 ANF buffers. Every interval includes input conversion/byte encoding, applicable resource preparation, fenced upload, encode/submit/wait, one explicit readback and reverse conversion. The second independent explicit read and all oracle comparisons are outside the timer. This is forward-only timing; device startup and destruction are excluded. Fresh_resources uses a warm device/compiler/driver context, not cold process initialization.

## Descriptive results and interpretation

Every one of the 360 process/panel median paired ratios is strictly above the prospective 1.05 threshold. Across those 360 medians, the range is **1.486527–12.467563**. This is not a confirmatory p-value, confidence bound, universal winner or SML throughput result.

These large end-to-end ratios must not replace the earlier resident-wall ratios: the endpoint and cost boundary changed. The present vec4 carrier performs scalar bit-matrix transposes while packed7's canonical gate-major carrier avoids them. For K=16384, G=2048, boundary, prepared, the median of 200 individual vec4 fractions `(conversion_in_ns + conversion_out_ns) / wall_ns` is **0.9637916664**. This is a per-sample fraction, not a sum of component medians. It shows that the current scalar conversion path dominates this measured endpoint. A future qualified SIMD transpose could change this comparison.

The current SciRust PVP conversions were also inspected at `16a88a55e1cf0e866f5405871b7f0e6d7cbd12bf:scirust-simd/src/pvp.rs`: those two conversion methods use scalar address/gate loops. No claim is made about every SciRust transpose primitive or an optimal SIMD baseline. No SciRust or SML default was changed.

All regressions/jitter are retained in the raw samples and p95/component tables in analysis.json; no slower individual sample was removed. p50 columns below pool the 200 measured values per endpoint descriptively. The ratio column is the median of ten process medians of twenty paired ratios; it is not a quotient of the p50 columns.

| K | G | Bank | Mode | vec4 p50 ms | packed7 p50 ms | Median of 10 paired process medians | Process-median range |
|---:|---:|---|---|---:|---:|---:|---:|
| 4096 | 128 | boundary | prepared | 1.240395 | 0.308592 | 3.905714 | 2.780551–4.935274 |
| 4096 | 128 | boundary | fresh_resources | 1.496537 | 0.834322 | 1.773646 | 1.643593–1.863566 |
| 4096 | 128 | sparse4 | prepared | 1.290815 | 0.326948 | 3.709188 | 2.507625–4.354978 |
| 4096 | 128 | sparse4 | fresh_resources | 1.443273 | 0.888863 | 1.673008 | 1.486527–1.747696 |
| 4096 | 128 | dense32 | prepared | 1.383611 | 0.309643 | 4.489604 | 4.329397–5.505366 |
| 4096 | 128 | dense32 | fresh_resources | 1.623004 | 0.847891 | 1.912201 | 1.797014–1.960565 |
| 16384 | 2048 | boundary | prepared | 72.491057 | 7.052198 | 10.473509 | 9.744797–12.467563 |
| 16384 | 2048 | boundary | fresh_resources | 74.655798 | 9.082280 | 8.305174 | 7.089735–8.581740 |
| 16384 | 2048 | sparse4 | prepared | 71.335848 | 6.578477 | 10.559660 | 10.217988–11.061581 |
| 16384 | 2048 | sparse4 | fresh_resources | 71.895438 | 7.425470 | 9.582037 | 9.337544–9.916397 |
| 16384 | 2048 | dense32 | prepared | 74.844398 | 6.467880 | 11.381255 | 10.921497–11.769888 |
| 16384 | 2048 | dense32 | fresh_resources | 75.503749 | 7.349965 | 10.211204 | 9.855064–10.450629 |
| 65536 | 512 | boundary | prepared | 71.267505 | 6.534863 | 10.871506 | 10.310031–10.979553 |
| 65536 | 512 | boundary | fresh_resources | 72.414283 | 7.472775 | 9.704720 | 8.970943–9.906337 |
| 65536 | 512 | sparse4 | prepared | 72.005371 | 6.592055 | 10.760877 | 9.928238–11.275278 |
| 65536 | 512 | sparse4 | fresh_resources | 72.654168 | 7.614350 | 9.555719 | 9.066400–9.864584 |
| 65536 | 512 | dense32 | prepared | 72.365110 | 6.577098 | 10.900118 | 10.302762–11.277629 |
| 65536 | 512 | dense32 | fresh_resources | 72.814143 | 7.517714 | 9.640994 | 9.200441–9.905128 |
| 262144 | 128 | boundary | prepared | 56.772782 | 6.732972 | 8.342775 | 6.753326–8.514673 |
| 262144 | 128 | boundary | fresh_resources | 57.711927 | 7.573640 | 7.449150 | 6.005371–7.768146 |
| 262144 | 128 | sparse4 | prepared | 53.238842 | 6.702269 | 7.912747 | 5.418774–8.065957 |
| 262144 | 128 | sparse4 | fresh_resources | 53.781570 | 7.658335 | 6.945593 | 5.642607–7.219878 |
| 262144 | 128 | dense32 | prepared | 55.274061 | 7.015929 | 7.792139 | 6.622407–8.436221 |
| 262144 | 128 | dense32 | fresh_resources | 55.905057 | 7.666747 | 7.153897 | 6.061522–7.509649 |
| 16384 | 129 | boundary | prepared | 6.772422 | 0.766071 | 8.046761 | 6.334421–10.596047 |
| 16384 | 129 | boundary | fresh_resources | 7.559418 | 1.966373 | 3.922352 | 3.128322–4.222050 |
| 16384 | 129 | sparse4 | prepared | 5.777776 | 0.824821 | 6.442202 | 6.065315–7.080404 |
| 16384 | 129 | sparse4 | fresh_resources | 5.091273 | 1.429047 | 3.513402 | 2.635956–4.154210 |
| 16384 | 129 | dense32 | prepared | 4.774507 | 0.739772 | 6.478428 | 4.775689–7.014160 |
| 16384 | 129 | dense32 | fresh_resources | 4.780771 | 1.306305 | 3.686613 | 3.558833–3.872317 |
| 65536 | 129 | boundary | prepared | 15.564585 | 1.392481 | 11.384000 | 10.670064–11.739345 |
| 65536 | 129 | boundary | fresh_resources | 16.546960 | 2.162955 | 7.508419 | 6.864064–7.793462 |
| 65536 | 129 | sparse4 | prepared | 15.464818 | 1.400991 | 10.857298 | 8.471160–11.390972 |
| 65536 | 129 | sparse4 | fresh_resources | 15.900367 | 2.174399 | 7.118545 | 6.233977–7.478002 |
| 65536 | 129 | dense32 | prepared | 16.065548 | 1.407207 | 11.466279 | 9.900985–11.674402 |
| 65536 | 129 | dense32 | fresh_resources | 16.536171 | 2.177812 | 7.491167 | 7.254347–7.733171 |

## Admission observations and the retained interruption

| Run | Scans | Partial scans | Unreadable entries | Incomplete markers | Foreign-user rejects | Final restoration |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 46 | 1 | 1 | 1 | 0 | verified |
| 2 | 46 | 1 | 1 | 1 | 0 | verified |
| 3 | 47 | 1 | 1 | 1 | 0 | verified |
| 4 | 47 | 0 | 0 | 0 | 0 | verified |
| 5 | 46 | 1 | 2 | 1 | 0 | verified |
| 6 | 41 | 9 | 13 | 9 | 0 | verified |
| 7 | 47 | 0 | 0 | 32 | 32 | verified after retained initial failure |
| 8 | 47 | 3 | 5 | 2 | 0 | verified |
| 9 | 46 | 3 | 3 | 2 | 0 | verified |
| 10 | 46 | 1 | 1 | 1 | 0 | verified |

Only run 4 passed every sampled observation decision with zero gaps and no foreign visible users. Even that weaker observation does not prove continuous device idle/exclusivity or close independent admission review. Run 7 has complete scans but positive foreign descriptor observations, so complete enumeration alone is insufficient. Other runs retain unknown/foreign gaps; zero accepted performance confirmations remain the correct verdict.

An avoidable coordination mistake occurred: merging PR #349 during the active campaign triggered legacy PVP3b CI on Thor. [Workflow 37618145051, job 112781410607](https://github.com/Memorithm/FLAT-ATTENTION/actions/runs/37618145051/job/112781410607) opened the same device and waited for its cooperative lock. Its log timestamps show the lock step at 12:02:56.9725217 UTC, subsequent inventory output at 12:03:33.5503804 UTC, and deferred_gpu_busy at 12:03:39.1096252 UTC. It started no PVP3b benchmark. These logs establish CI overlap and waiting; they do not attribute every foreign PID/gap or establish concurrent GPU compute.

After process 7 had completed exactly and the controller restored its services/policies, the separate restoration inspector failed the nonblocking lock-availability assertion. The first inspector stderr and orchestrator exit 1 are retained. A later **read-only** reinspection at 12:04:47.790239751 UTC passed; no earlier GPU process was rerun. A frozen resume script then executed only the already-planned processes 8–10. This reconciliation remains part of the diagnostic cohort and does not repair admission for process 7.

The accompanying CI correction changes the legacy GPU step from blocking flock to immediate nonblocking deferral when the cooperative reservation is held, before GPU inventory/benchmark. It prevents that workflow from waiting with a device descriptor for minutes. The busy/free lock controls and all six embedded shell blocks pass local checks. It does not create an idle/exclusivity capability or remove the legacy workflow's vendor-management dependency. Future native campaigns must finish and restore before a merge or new CI dispatch.

## Evidence and independent integrity checks

Archive: `evidence/pvp-canonical-e2e-thor-20261007/capture.tar.xz`.
SHA256: `5e9adafca249ed181945d40ff8b52c8c804b76231ff516b3c0d94b9ab97fc4af`; bytes: 896960.
Git blob: `3d3e70dba706ea219fd002208e12c48eb157dee0`.

The archive retains every raw log, observation, lifecycle trace, controller, frozen input hash, service receipt, initial failed restoration, reconciliation, native build record, overlapping CI log and analysis parser erratum. Original historical binaries remain unchanged. Final process-10 restoration was inspected at 2026-10-07T12:09:15,983124636+00:00.

An independent local archive read verified 337 declared file hashes and 12 frozen input hashes, then reexecuted the strict analyzer and obtained the same ten-process coverage and zero performance admission. Six mutations of temporary copies—wrong source, duplicate sample, missing check, wrong word, wrong word count, zero wall time—were rejected. No native GPU test was repeated for those checks.

The first strict analysis rejected the source line because libtest prefixes its first printed record with the test name. The correction accepts only the exact expected test-name prefix followed by the protocol marker. The earlier rejected analysis and its analyzer bytes are retained; the raw timing logs were not edited.

## Remaining promotion gates

An independently qualified/reviewed admission and session-sampling method remains absent, and this cohort contains rejected observations and an interruption. Therefore the frozen conditional sign-test rule is not activated, no confirmatory p-value is emitted and accepted performance rows stay zero. Cost coverage and bounded forward correctness have advanced; independent performance admission and any SML-owned model/destination qualification have not closed.
