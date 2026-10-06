# PVP full-grid explicit-readback control — 7 October 2026

This separate prospective correctness campaign follows the six-process paired
diagnosis. Its automatic processes all failed and its three explicit-transition
processes passed. This new campaign is not a replacement of R11 or a proof of
an upstream root cause.

Freeze before execution: ten fresh processes, all running the new opt-in
`pvp3d_three_candidate_bench --measure-map-transition` mode. Use the exact
compiled/run source SHA and binary SHA256. Preserve the historical full grid,
fixture, candidates, six permutations, five warmups, thirty repetitions,
pre/post full-state comparisons, capability rejections and timer interval.
The only new control is COPY_DST-to-MAP_READ transition of each correctness
readback before queue submission. Default `--measure` remains automatic.

Execute under one independently acquired/released cooperative Thor reservation,
with prearmed independent restoration, original policies, temporary guards and
RemoteOps observations retained. Ten complete processes are the fixed count;
do not replace failures. A process timeout remains an incomplete failure.
Continue diagnostic executions after unknown observations and retain them as
unknown. No accepted performance-admission rows are produced in this campaign.

Report the exact number of complete pre/post oracle passes, failures and unknown
observations. Even 10/10 passes control only the tested source/device/workload
and pre/post states; they do not verify every intermediate measured state or
prove a failure probability of zero. Timing rows remain diagnostic, distinct
from historical R11 and from future prepared-plan/layout performance trials.

The new packed-address kernel is not run by this example. Its separate ANF
functional qualification does not resolve the mapping defect or qualify speed.
