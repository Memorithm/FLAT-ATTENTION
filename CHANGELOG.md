# Changelog

All notable FLAT-ATTENTION changes are recorded here. The project does not treat a merged milestone as a released semantic version: a release is created only after the release checklist and exact-head qualification gates are satisfied.

## Unreleased — 1.0 candidate

### Documentation and project clarity

- Corrected the README licensing section: the repository is source-available under PolyForm Noncommercial 1.0.0 (`LICENSE` / `LICENSE.md` / `LICENSING.md`), the same license as SciRust. The previous "No license grant is declared" wording was outdated and incorrect.
- Added README badges (CI, Rust 1.89, PolyForm Noncommercial) aligned with SciRust wording.
- Added README sections for **Project status**, **Repository layout**, and a **Quickstart (host-only)** path so newcomers can orient themselves without reading the full technical architecture first.
- Added `examples/hello_attention.rs`, a default-feature scalar-oracle smoke example that requires no GPU.
- Added `crates/README.md` and `docs/CRATES.md` so research/candidate crates are not mistaken for the reusable `api::v1` contract.
- Pointed the docs index at `ROADMAP_STATUS.md`, the crate map, and the corrected licensing files for first-time readers.

### Kernel compilation platform
