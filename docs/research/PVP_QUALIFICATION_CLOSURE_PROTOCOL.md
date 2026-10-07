# PVP qualification closure — prospective gate probe (2026-10-07)

Status: admission diagnosis only. No final performance validation is currently justified.

## Defined claims

Correctness is qualified for the published finite ANF corpus and native Thor configurations, not for every possible model or machine. A performance conclusion must name its adapter, driver, source, geometry, bank family, endpoint and control. Physical register allocation is unmeasured.

Historical PR #347 contains 10,800 resident-wall samples and 60,750 exact comparisons, but all three processes have unknown occupancy. PR #348 preserves that verdict. No historical timing is promoted by this protocol.

## First action: close the admission blocker

Run the committed admission-only controller on Thor. It retains the existing frozen e04c980817cac889e759fb2bb33cf231ef737e73 source and executable identity, but starts no benchmark. Its cooperative device lock, scoped pause of the four already-authorized services, independent restoration timer and five preflight scans are unchanged. Only the namespace and early stop differ from the published timing controller.

The observer is the frozen Rust v2 binary (SHA256 2750a44e6167875081368cd7df6b7ce582e77df17473ce5685727659035149fd). Foreign/unknown gaps, truncated observations, foreign visible users and trace loss reject the gate. Limited preflight retries are retained as diagnosis; every attempt is kept. Process exits do not prove lack of GPU use. A clean preflight still does not qualify continuous exclusive access.

No benchmark runs even if every preflight passes. This probe cannot establish hardware idle state or replace independent admission review. The controller's exit 2 means rejected observation after retries, not a test failure or an admissible performance result. Restoration is checked independently after the controller exits, including original service policies, removed guards/timer/trace, available cooperative lock and unchanged binaries.

## Required closure evidence

1. Independently reviewed vendor-neutral admission and contamination control, with exact implementation/evidence identity and an explicit trust/scope statement. Sampled descriptor scans alone are insufficient.
2. Fresh prospectively frozen correctness and timing campaign on an identified hardware adapter. Retain every failed process, geometry, regression and unknown observation.
3. Repeatability at the independent process/session level. Repetitions within one process are paired observations, not thousands of independent experiments. Freeze the candidate, endpoints, minimum meaningful improvement, session count, statistical rule and multiplicity handling before observing the confirmatory cohort.
4. Separate resident encode/submit/wait, optional public GPU timestamps and complete conversion/preparation/upload/compute/readback cost. Report amortization only for frozen reuse counts, including a one-use case. No resident win implies an end-to-end win.
5. SML-owned model-aligned ANF-bank reference and destination requalification for any SML integration claim. Synthetic kernel evidence does not close this gate.

A kernel claim may close on a defined synthetic corpus without claiming SML model throughput. SML final validation requires all applicable model and destination gates. CI green permits an engineering merge; it does not establish any of these scientific claims.

## Run identity and safety

Freeze controller/restoration/checker bytes and hashes before execution in a new campaign directory. Use a distinct systemd scope, RuntimeMaxSec=25min and a 30-minute independent restoration timer. Read each preparation result before dispatch. Do not modify historical campaign directories or binaries. Do not pause additional host services, bypass the admission checker, introduce a vendor SDK or retry an ambiguous side effect without reconciliation.

This is one bounded gate probe, not repeated sampling until a favorable verdict. If rejected, publish the actual blocker and restoration receipt; do not relabel the result as final validation.
