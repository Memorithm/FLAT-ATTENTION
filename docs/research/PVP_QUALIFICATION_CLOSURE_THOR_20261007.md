# Thor PVP qualification closure probe — 2026-10-07

Verdict: the bounded preflight observation gate passes; final performance qualification remains open.

## Frozen action and observed result

Protocol/controller source: 37841d2789004b38106ac5a58e7cd2df73d8313f.
Historical timing source: e04c980817cac889e759fb2bb33cf231ef737e73, unchanged.

On Thor, the controller reserved the cooperative device lock at 11:45:30.552972753 UTC, temporarily paused the four already-authorized services and performed five consecutive preflight scans. Every scan returned observed with zero unreadable entries and zero gaps. The sole visible user was PID 3070129, the controller holding the device descriptor for the cooperative flock. The live checker accepted its qualification-cgroup ownership.

No benchmark was started. The service finished successfully, runtime 9.563 seconds, and restoration completed at 11:45:40.014569823 UTC. The lifecycle trace retains 150 records; every recorded CPU trace-loss counter is zero.

The later independent read-only restoration inspection (11:46:47.960583855 UTC) verified all four services active with their original Restart/RefuseManualStart policies, removed guards/trace instance, inactive restoration timer, available lock and unchanged historical/timing binaries.

The first ad-hoc read-only inspection incorrectly expected an empty users array during reservation and failed that assertion after the restoration checks passed. The controller's lock descriptor is expected. That inspection error and its correction remain in the archive; no host action or measurement was repeated to replace a failure.

## What this closes

The exact admission-only code was frozen before execution. A scoped service pause can produce five complete, controlled preflight observations on Thor. Native controller exit and restoration succeeded. This is concrete admission-path diagnosis.

It does not establish continuous GPU inactivity, hostile-process fencing, independence from host scheduling, or an independently qualified exclusive lease. It does not repair the unknown observations in the earlier timing campaign. No fresh latency samples, admitted performance rows, speed claim, register allocation claim or SML model validation were produced.

Final closure still requires the gates in [the prospective closure protocol](PVP_QUALIFICATION_CLOSURE_PROTOCOL.md): independently reviewed continuous admission, a fresh frozen repeatability study, complete conversion/transfer/preparation costs and separate SML destination qualification where that claim is intended.

## Evidence

Archive: `evidence/pvp-admission-thor-20261007/capture.tar.gz`.
SHA256: `6e52185b99a70bc9a87bc6bdb6107a8759c94feca0911a0eb16d1b21b6e8293a`.
Bytes: 12189.

The archive retains controllers, frozen input identities, every scan, per-scan trace statistics, complete lifecycle trace, service state receipts, dispatch output, inspection output, erratum and per-file hash manifest. An independent local read of the downloaded archive verified all 33 declared file hashes, all eight frozen input hashes, five exact observation records, trace-loss counters and zero timing logs. This checks record integrity and consistency, not reexecution or exclusive hardware access.
