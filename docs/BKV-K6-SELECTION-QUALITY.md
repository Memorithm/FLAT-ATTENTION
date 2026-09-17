# BKV-K6 declared-target selection quality evidence

Status: research-only evidence contract. No runtime-routing or promotion change.

BKV-K6 needs a real quality/recall gate for approximate Boolean page selection before any adaptive BKV-7 policy can consume it. Timing, candidate density, and logical K/V bytes avoided cannot establish that the router retained the pages a dense reference says matter.

`api::bkv6_selection_quality` adds the versioned `flat.bikv-selection-quality.v1` record. The caller must provide a non-empty, strictly increasing, duplicate-free set of logical page IDs derived independently from the declared dense-reference target for that experiment. FLAT does not construct or tune this target set.

The record first revalidates the exact `flat.boolean-kv-selection.v2` router evidence, then retains that canonical selection beside the declared target and computes exact integer counts for true-positive pages, false-negative pages, false-positive pages, target-page recall, false-negative rate, and candidate density. Ratios are retained canonically as numerator/denominator pairs; floating-point accessors are convenience views only.

The contract rejects an empty/undefined dense target, an empty mapped-page universe, out-of-range target pages, duplicates, non-monotone target order, and any already-invalid Boolean selection. An empty Boolean selection is valid evidence when the underlying selection accounting is valid; against a non-empty target it records zero recall and complete false-negative rate rather than disappearing.

This is **target-page selection quality**, not downstream model quality. The experiment must separately preregister how the dense target is produced, the recall/FNR acceptance threshold, task/model quality metrics when applicable, and whether the run is development, validation, or confirmatory. The threshold is deliberately not embedded in this API so a caller cannot mistake a library default for a preregistered scientific decision.

This evidence does not show physical DRAM/HBM traffic avoidance, latency improvement, TTFT/TPOT improvement, preserved perplexity/task quality, or readiness for BKV-7. Those require the independent measurements already required by the BKV-K6 qualification gate and Boolean-KV roadmap.
