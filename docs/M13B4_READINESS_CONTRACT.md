# M13B.4 first-token readiness contract

Status: research-only correctness gate. No overlap, latency, bandwidth, throughput, or quality result is claimed by this document.

This slice implements the first non-timing part of the preregistered M13B.4 protocol in `api::research_boolean_overlap`.

Before a first decode invocation may be described as **first-token ready**, the backend-neutral snapshot must establish:

1. numerical K/V metadata and Boolean K/KV metadata use the same generation;
2. the numerical page mapping covers exactly the committed prefill prefix;
3. Boolean signature metadata covers exactly the same prefix at the same page geometry;
4. the first decode invocation consumes that exact generation and prefix;
5. the first decode critical path performs zero historical K/KV signature rebuilds.

For a committed prefix of `T` tokens and page size `P > 0`, the expected metadata page count is exactly:

```text
ceil(T / P)
```

A partial final page counts as one page. Zero-length prefixes are well-defined and require zero pages.

The contract fails closed on generation drift, page-coverage drift, arithmetic overflow, zero page size, consumer prefix drift, and historical-signature rebuilds.

## What this proves

Passing the gate proves only that the first decode invocation can consume already-committed Boolean routing metadata for the complete declared prefix without reconstructing historical signature metadata on that invocation.

## What this does not prove

Passing this gate does **not** prove:

- Boolean and numerical execution overlap in time;
- lower first-token or steady-state latency;
- fewer physical DRAM transactions;
- lower energy use;
- preserved downstream model quality;
- safety of any particular learned routing policy;
- production readiness.

Those claims remain subject to the timing, trace, quality, hardware, and matched-baseline requirements frozen in `docs/M13B4_PREREGISTRATION.md`.

## Next implementation gate

The next M13B.4 slice is a versioned trace/evidence contract for the preregistered events and scheduling variants. Only after that instrumentation is qualified should multi-dispatch overlap timing be interpreted.
