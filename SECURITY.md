# Security Policy

## Supported state

FLAT-ATTENTION is pre-1.0 (`0.1.0`, `publish = false`). Only the default
branch receives security fixes.

## Reporting a vulnerability

Do **not** open a public GitHub issue for suspected vulnerabilities.

Report privately through one of:

1. **GitHub Security Advisories** (preferred): *Security* tab of this
   repository → *Report a vulnerability*.
2. **Email**: the maintainers listed in `Cargo.toml` (`authors`), subject
   prefixed `[security]`.

Include: affected commit SHA, a minimal reproducer, and the impact you
observe. You will receive an acknowledgement within **7 days**.

## Coordination

- Findings are triaged against the current default branch.
- Fixes land without public discussion first; a release note follows once a
  patched revision is qualified per `docs/RELEASE_CHECKLIST.md`.
- Reporters who wish to be credited in the advisory should say so in the
  initial report.

## Scope notes

This crate is an in-process compute library. Areas of highest security
interest for reporters:

- **Host-side validation**: shapes, lengths, and device limits are validated
  before allocation or dispatch; violations must surface as typed errors,
  never as panics, oversized allocations, or wgpu uncaptured-error aborts.
- **Shader/host layout agreement**: WGSL uniform/storage layouts must match
  the host encoders field-for-field (see `shaders/` and the `wgpu_*`
  modules).
- **Supply chain**: dependency advisories and license policy are enforced by
  `.github/workflows/supply-chain.yml` (`cargo-deny`); actions are pinned by
  commit SHA. The crate itself declares `#![forbid(unsafe_code)]`.

Out of scope: the operating-system/driver boundary below wgpu, and the
self-hosted Thor qualification runners (documented trust model in
`.github/workflows/*thor*.yml`).
