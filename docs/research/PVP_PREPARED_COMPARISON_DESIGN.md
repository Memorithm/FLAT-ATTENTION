# Prepared-plan comparison design

Status: design only, not an executed benchmark or final preregistration.
Implement and freeze the measurement harness after the large-domain
correctness gate has passed and its pull request has merged.

Compare equivalent ANF banks and logical transforms using prepared plans
for the existing vec4, fused2 and tile8 candidates, a gate-major single-
stage control, and the address-packed seven-stage-prefix candidate.
Separate layout effects from stage fusion; dispatch counts alone do not
prove a performance gain. No candidate becomes the default automatically.

Report pipeline creation, layout conversion, plan preparation and upload
separately. Report resident encode/submit/completion costs separately from
end-to-end costs including conversion and transfer. Optional public WGPU
timestamps, when supported, form a distinct cohort from CPU wall time.
Do not compare unlike timing boundaries or silently mix capabilities.

Verify every measured state against an independent logical oracle, not
only a pre/post smoke test. Freeze geometries, banks, repetitions,
ordering, warmups, exclusion rules, sources and binaries before execution.
Retain every failure and reject the affected cohort when a required
capability, identity, correctness or occupancy gate is missing. Preserve
raw paired measurements and uncertainty; report regressions as well as
gains. Final counts and statistical criteria belong to the future frozen
harness protocol, not this design note.
