# BKV-K6 — Boolean-selected paged numerical decode

Status: research-only correctness gate. No production routing or performance claim.

## Purpose

BKV-K5 established a host-side BIKV handoff from Boolean page selection to authoritative numerical paged K/V. BKV-K6 adds the first numerical consumer that can execute a non-contiguous selected page set without densely renumbering the surviving pages.

The existing M16 paged decode remains the qualified dense baseline and is not modified by this experiment.

## Invariant

Each selected descriptor carries:

```text
physical_page
original logical_page
live_tokens
```

The original logical page is preserved even when selected pages are sparse. K is already RoPE-rotated at its original token position, and the BKV-K6 shader derives the logical key position from `logical_page * page_size + offset_in_page` rather than from a compacted survivor index.

Numerical K/V remains authoritative. The Boolean tier selects which pages may be touched; it does not reconstruct numerical K/V.

## Fail-closed host validation

Before encoding device work, the selected table is revalidated against the current `PagedKvTable`:

- Boolean selection generation equals numerical KV generation;
- full live-token and mapped-page snapshot matches;
- at least one numerical survivor page exists;
- selected logical pages are strictly increasing and unique;
- every logical page is in range;
- every selected physical page equals the authoritative numerical mapping;
- every selected page carries the exact authoritative live-token count, including the final partial page;
- selected page count stays within the portable bounded uniform-table capacity.

A stale or forged selection must fail before GPU submission.

## Correctness gates

BKV-K6 is accepted only when all of the following hold:

1. The WGSL source parses and validates under Naga.
2. An all-accept BIKV selection matches the existing contiguous numerical attention oracle for O and LSE.
3. A non-contiguous selection matches a restricted numerical oracle that rotates K at the original token positions; compact survivor renumbering is not permitted.
4. Forged physical-page metadata is rejected.
5. Stale generation metadata is rejected.
6. Existing M16 dense paged decode remains untouched and green.

## Explicit non-claims

BKV-K6 does not claim:

- end-to-end attention speedup;
- reduced physical DRAM traffic solely from source structure;
- quality preservation for approximate Boolean selection policies;
- automatic production routing;
- superiority over dense M16.

Performance qualification comes only after correctness is green and must include Boolean signature creation/search, synchronization, selected numerical work, and any transfer or staging cost.

The promotion condition remains:

```text
T_boolean_selection + T_selected_numerical_attention < T_dense_numerical_attention
```

under a declared quality/recall gate and with a dense fallback.
