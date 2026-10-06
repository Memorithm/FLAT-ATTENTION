# IND1 operator repetition on Thor — 6 October 2026

## Result

**Gain confirmation is rejected.** This is an independent operator repetition on the **same Thor GPU** already used for R8–R11, not a distinct device. Dell was not measured.

Block 1 of the frozen ten-process battery produced five complete, exact processes. Block 2 never started. A second attempt (IND1B) aborted at occupancy admission, before ANF and before timing. Occupancy on block 1 was not accepted for the whole battery. **Accepted confirmation rows: zero.** No configuration meets the frozen ten-process engineering rule. No SML-GENIUS speedup, default route, autotune, or model-quality claim follows from these numbers.

The intermittent large-width oracle failure (R11: vec4, K=262144, G=128, 16320 wrong u32 words) was **not observed** in these five processes. That does not show it is gone.

A separate shared-device run of `pvp_transfer_reuse` on merge `bff0fa4` passed 40 rounds. That success does not fix or explain PVP.

## What was executed

| Id | GitHub Actions run | UTC window | Role | Outcome |
|---|---|---|---|---|
| Build | [37546137186](https://github.com/Memorithm/RemoteOps/actions/runs/37546137186) | before reservation | `cargo +1.89.0 build --locked --release --features wgpu` of historical `ddb6f01` | success; binary hashes match the published Thor hashes |
| IND1 block 1 | [37546438832](https://github.com/Memorithm/RemoteOps/actions/runs/37546438832) | 2026-10-06T23:25:11Z–23:25:36Z | five fresh `--measure` processes, one reservation | functional complete; performance admission rejected; launcher exit 1 before block 2 |
| IND1B | [37546748572](https://github.com/Memorithm/RemoteOps/actions/runs/37546748572) | 2026-10-06T23:28:30Z–23:28:44Z | new id, not a rerun of failed process numbers | admission rejected before the bench; no timings |
| Transfer | [37547728094](https://github.com/Memorithm/RemoteOps/actions/runs/37547728094) | 2026-10-06T23:39:20Z–23:40:00Z | merge `bff0fa4`, shader-free 4 MiB copies, services left running, no exclusive flock | 1 test passed |

Protocol text was frozen before measurement at RemoteOps `research/pvp-ind1-2026-10-07` commit `6ab15b0e94cd13359116be524f165e24067120ca`. The historical bench source was not edited. No failed result was deleted or replaced under the same process number.

## Identity

| Item | Value |
|---|---|
| Execution source | `ddb6f0144821dfa29d268601e7087e5001278e52` |
| Evidence merge (not the measured bench) | `bff0fa4f74f86c64073c376a11c60b6d5b227c49` |
| Host | `tarek`, aarch64 |
| Adapter | NVIDIA Tegra NVIDIA Thor |
| Backend | Vulkan |
| Device type | IntegratedGpu |
| Vendor / device | 4318 / 11008 (`0x10de` / `0x2b00`) |
| Driver | NVIDIA 580.00 |
| Compiler | rustc 1.89.0 (29483883e 2025-08-04) |
| Bench SHA256 | `7aef67026ad088504478885cf9f2b7c864218f6fb3b0792bd8609fd907d3f414` |
| ANF harness SHA256 | `4458f8e245476cb5ec65ace1b8ab57f09bc7a9e4e137a4e70328b9652987adfa` |
| Observer SHA256 | `2750a44e6167875081368cd7df6b7ce582e77df17473ce5685727659035149fd` (pinned RemoteOps `7cc8203e`, not rebuilt) |
| Timing scope printed by the bench | resident encode + queue submit + device wait |
| Order / warmups / repeats | ABC/ACB/BAC/BCA/CAB/CBA, 5 warmups, 30 repeats |
| Performance claim printed by the bench | `none` |

The controller checked those three SHA256 values before the processes. `FLAT_SOURCE_REVISION` at compile and run was the historical SHA. Numeric WGPU limits are **not** printed by this frozen bench. Je ne sais pas their numeric values. The observable limit behavior is `skipped_limit` / `cohort_dispatch_limit`.

Software adapter llvmpipe was present on the host and was not used. `nvidia-smi` was not used.

## Functional control

ANF, once, before timing, with `WGPU_BACKEND=vulkan`, `FLAT_REQUIRE_WGPU=1`:

`PVP_ANF_BANK_COMPLETE,cases=24,candidate_checks=96,performance_claim=none`

Every archived case line in that log is `truth=exact,round_trip=exact`. This is the same 24-case synthetic bank as R8/D2 (96 candidate checks), not R11's two-block 192, and it does not cover the large-width benchmark or SML semantics.

Five benchmark processes each printed `qualification_status=complete` and `run_N_oracle=pass`. There is no `PVP3d` mismatch line and no mismatched-word count in run [37546438832](https://github.com/Memorithm/RemoteOps/actions/runs/37546438832).

For every measured `(K,G,candidate)`, the input checksum and the output checksum are identical across all five processes. For every measured `(K,G)`, vec4, fused2, and tile8 share one output checksum. That is exactness of these five invocations, not a proof that the intermittent failure cannot recur.

Grid from the frozen protocol: K = 256, 1024, 4096, 16384, 65536, 262144, 1048576; G = 128, 512, 2048. Each process emitted 45 measured rows and 18 skipped rows.

Skipped in every process, reason `cohort_dispatch_limit`:

| K | G |
|---:|---:|
| 65536 | 2048 |
| 262144 | 512 |
| 262144 | 2048 |
| 1048576 | 128 |
| 1048576 | 512 |
| 1048576 | 2048 |

Timed diagnostic samples retained: 225 rows × 30 repeats = **6750**. Warmups are outside the percentiles. The frozen bench does not print the 30 individual samples, so per-sample dispersion below p50/p95 is not available. Je ne sais pas the individual sample values.

## Diagnostic latency (not qualified)

Reduction = `1 - p50_candidate / p50_vec4` **inside each process**, then the **median of those five ratios**. This is not the ratio of the median p50s. Absolute p50/p95 columns below are the median of the five process-level percentiles, in nanoseconds, and are not themselves the gain metric.

`≥5%` counts how many of the five processes had a reduction of at least 0.05. The frozen rule needs nine of **ten** processes and both reservation blocks. Five of five is not that rule.

| K | G | n | vec4 p50 | vec4 p95 | fused2 p50 | fused2 p95 | tile8 p50 | tile8 p95 | fused2 | tile8 | fused2 ≥5% | tile8 ≥5% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 128 | 5 | 231796 | 268195 | 208695 | 239731 | 185610 | 225269 | +9.75% | +19.45% | 5/5 | 5/5 |
| 256 | 512 | 5 | 249852 | 266620 | 225166 | 282102 | 201481 | 226055 | +9.77% | +19.96% | 5/5 | 5/5 |
| 256 | 2048 | 5 | 278982 | 300064 | 247749 | 272361 | 227852 | 264610 | +9.51% | +17.04% | 5/5 | 5/5 |
| 1024 | 128 | 5 | 370268 | 409732 | 344380 | 370832 | 311166 | 339537 | +6.99% | +16.84% | 5/5 | 5/5 |
| 1024 | 512 | 5 | 407037 | 440629 | 373101 | 388861 | 343491 | 359046 | +8.34% | +15.68% | 5/5 | 5/5 |
| 1024 | 2048 | 5 | 413694 | 428453 | 379620 | 392555 | 361629 | 396102 | +8.24% | +12.38% | 5/5 | 5/5 |
| 4096 | 128 | 5 | 478304 | 493055 | 445138 | 462480 | 414935 | 425749 | +6.87% | +13.16% | 5/5 | 5/5 |
| 4096 | 512 | 5 | 431767 | 494509 | 402536 | 468221 | 407722 | 444870 | +6.77% | +9.98% | 5/5 | 4/5 |
| 4096 | 2048 | 5 | 534250 | 572416 | 501684 | 528323 | 533175 | 559924 | +6.63% | +0.20% | 5/5 | 0/5 |
| 16384 | 128 | 5 | 466055 | 589519 | 442943 | 539360 | 422852 | 472009 | +5.57% | +9.27% | 4/5 | 5/5 |
| 16384 | 512 | 5 | 579351 | 669369 | 546166 | 615499 | 570185 | 631878 | +5.73% | +1.32% | 4/5 | 0/5 |
| 16384 | 2048 | 5 | 760026 | 861369 | 717471 | 807582 | 924629 | 1006229 | +5.53% | −19.18% | 5/5 | 0/5 |
| 65536 | 128 | 5 | 624342 | 724963 | 608860 | 695119 | 621019 | 706850 | +3.60% | +2.33% | 0/5 | 0/5 |
| 65536 | 512 | 5 | 942045 | 997777 | 908026 | 948749 | 1097814 | 1138721 | +3.82% | −15.46% | 0/5 | 0/5 |
| 262144 | 128 | 5 | 1012785 | 1102517 | 970026 | 1095118 | 1073341 | 1230276 | +0.84% | −11.45% | 0/5 | 0/5 |

Slowdowns and near-zeros, still diagnostic:

- tile8 at K=16384 G=2048: **−19.18%** (every process slower; worst process −21.66%).
- tile8 at K=65536 G=512: **−15.46%**.
- tile8 at K=262144 G=128: **−11.45%** (every process slower).
- tile8 at K=4096 G=2048: median of ratios **+0.20%**, with process ratios −2.25%, +2.01%, +0.20%, +1.84%, −3.14%. Not a gain.
- tile8 at K=4096 G=512 is +9.98% by the median of ratios, but one process is **−4.63%**.
- fused2 at K=262144 G=128 is +0.84% by the median of ratios. The five process ratios are −2.21%, −0.74%, +1.02%, +0.84%, +4.22%. The ratio of the median p50s is +4.22% and is **not** the metric.

There is no global winner. tile8 is faster than vec4 on the small geometries in this diagnostic and slower on the large ones listed above. fused2 stays non-negative at the median except that one large-K process is slower, and the median gain falls under 5% from K=65536 upward.

These medians sit near the previously published unqualified R11 diagnostic table (that table had ten cohorts, or nine for the last row). Nearness on the same binary and the same Thor is expected. It is not a new qualified confirmation and it is not pooled with R11 or with Dell.

## Why nothing is qualified

All of the following are true:

1. Only one reservation block ran. Five processes, not ten. The launcher exited 1 at `2026-10-06T23:25:36Z` because a non-blocking `flock` on `/dev/nvidia0` failed immediately after the systemd unit exited, even though `reservation_released_exit=0` and `block_1_systemd_rc=0`. Block 2 was not started. A later probe found the cooperative lock free. Whether that immediate failure was release latency or another holder is not proven for that instant.
2. During block 1 measurement the archived checker recorded `performance_admission_rejected=unknown_or_foreign_device_users_run_3`, `_run_4`, and `_run_5` (`admission_rejected=unknown_or_foreign_gap` then `foreign_visible_user`). Runs 1 and 2 have no such rejection line. One rejected run rejects the battery. The checker did not record the foreign cgroup text. Je ne sais pas which process those holders were.
3. The second flock during the reservation did fail while the first was held (`reservation_lock_contends=yes`), which is the required contention check, not a performance result.
4. Source and binary identity matched, and these five traces were not discarded. That is necessary and not sufficient.

IND1B (`37546748572`) took a new reservation, saw `admission_rejected=unknown_or_foreign_gap` on partial device-user observations, released with `reservation_released_exit=2`, and did not launch the bench. No performance numbers. It was not used to manufacture five or ten successes.

No further exclusive battery was started. Repeating until a clean occupancy window appears would be admission shopping, not this campaign.

## Intermittent error

Not reproduced here. K=262144 G=128 vec4 completed in all five IND1 processes with the same output checksum as fused2 and tile8. R11's 16320-word failure remains on the record. Cause of that failure: **unknown**. Stale-content patterns in R3–R5 remain an indication of stale returned data, not a proven driver defect. R5–R7 already showed that an upload wait or buffer retention which can pass a smaller test does not clear the full bench. No new discriminating one-change reproducer was built, because this repetition did not produce a mismatch to isolate.

## Shared transfer test

Separate checkout, merge `bff0fa4f74f86c64073c376a11c60b6d5b227c49`, not the historical bench tree.

Command, on Thor, rustc 1.89.0, lockfile unchanged:

`FLAT_REQUIRE_WGPU=1 WGPU_BACKEND=vulkan cargo +1.89.0 test --locked --release --features wgpu --test pvp_transfer_reuse -- --nocapture`

The cooperative lock was free at start and end. The four services stayed `active`. No exclusive flock was held. No `nvidia-smi`.

`PVP_TRANSFER_ADAPTER,name=NVIDIA Tegra NVIDIA Thor,backend=Vulkan,driver=580.00`

`PVP_TRANSFER_COMPLETE,rounds=40,words_per_round=1048576,all_words_exact=true,performance_claim=none`

Test binary SHA256 `3fa9d5e46fca4ccd054a7be38f6c67a69162c67163aff81a2bd81bf7314ff9d4` (`pvp_transfer_reuse-4cb73b2978d78789`). Cargo exit 0. Finished in 0.65s after a 34.16s release compile. This is a shared-device correctness check. It does not prove the PVP oracle failure is fixed.

## Restoration

IND1 and IND1B stopped `memorithm-clm`, `memorithm-clm-encoder`, `memorithm-viggle-lan`, and `rustdesk` only inside the reserved unit, after the runner was still the control path (`control_channel_after_stop=alive`; `control_channel_before_stop` was logged `unknown` because `systemd-run` did not inherit `GITHUB_RUN_ID`). Policies were restored by the unit cleanup. Two immediate CLM postflight runs failed their health assertion when checked about twelve seconds after restart. Later postflights passed, including [37546941501](https://github.com/Memorithm/RemoteOps/actions/runs/37546941501) and the post-transfer restore [37547751839](https://github.com/Memorithm/RemoteOps/actions/runs/37547751839) (`CLM_POSTFLIGHT_PASSED`, `LIVE_HEALTH` ok).

`ops/task.sh` on RemoteOps `main` `e6bc3b3f4185cf9318dd08013d99cd280597d70d` is byte-identical to the pre-campaign CLM postflight (`25e7f6ca`, SHA256 `6fc51619473e0ca26f35134f6a04d63c393e275a99b5b40144be8bffc91b3716`). Dell `ops/dell.sh` was not modified. Roadmaps on `agent/ecosystem-roadmap` were not merged. No FLAT source change, no fix PR, no CI merge.

## Comparison

| Campaign | Device | Exact complete processes | Confirmation |
|---|---|---:|---|
| R8 | Thor | 3 | not confirmatory |
| Dell D2 | RTX 4060, shared | 3 | diagnostic only |
| R11 | Thor | 9 of 10 | rejected (oracle and occupancy) |
| IND1 | same Thor | 5 of 10 requested | rejected (block 2 absent and occupancy) |
| IND1B | same Thor | 0 | rejected before timing |
| Transfer `bff0fa4` | same Thor, shared | 40 shader-free rounds | not a PVP result |

## Unproven for SML-GENIUS

- Any end-to-end model speedup. These are structural PVP kernel latencies.
- A qualified ≥5% gain on any geometry.
- Absence of the intermittent large-width failure.
- Which component returns stale words when the failure occurs.
- Physical GPU exclusivity. The cooperative flock is not that.
- Behavior on any GPU other than this Thor. Dell was not rerun.
- Numeric WGPU limits of this adapter, beyond the skip reason.

## Artifacts

Re-parsed from the Actions log of run 37546438832 (dispatcher prefix stripped at the first `Z `). Host copies under `/var/tmp/remoteops-pvp-ind1.37546438832` and `/var/tmp/remoteops-pvp-ind1b.37546748572` were not rewritten.

- [evidence/pvp3d-ind1-2026-10-07/measured.csv](evidence/pvp3d-ind1-2026-10-07/measured.csv)
- [evidence/pvp3d-ind1-2026-10-07/skipped.csv](evidence/pvp3d-ind1-2026-10-07/skipped.csv)
- [evidence/pvp3d-ind1-2026-10-07/median-of-ratios.json](evidence/pvp3d-ind1-2026-10-07/median-of-ratios.json)
- [evidence/pvp3d-ind1-2026-10-07/SHA256SUMS](evidence/pvp3d-ind1-2026-10-07/SHA256SUMS)
