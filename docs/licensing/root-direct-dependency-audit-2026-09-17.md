# Root direct dependency license audit — 2026-09-17

Scope: the four third-party dependencies declared directly by the root `flat-attention` package at source revision `27dc1790635d54b6ceb0bff654b42b20f60834ab`. This is not a repository-wide or transitive dependency audit.

The root `Cargo.lock` at that revision has SHA-256:

`7595fc78d944d5f3547c807b3e2a274bbdd66bb3bae8f09cacb73c170956b9a3`

`cargo metadata --locked --format-version 1` and the exact crates.io registry packages resolved by that lockfile report:

| Package | Exact resolved version | Cargo.lock crates.io checksum | Upstream package license metadata | Bundled license files inspected |
|---|---:|---|---|---|
| `wgpu` | `30.0.1` | `527ccdf43dd5b2e8676eed9984ce00e2bbb0a1b85b70c1969dcb6cd2eb55ab9e` | `MIT OR Apache-2.0` | `LICENSE.MIT`, `LICENSE.APACHE` |
| `pollster` | `1.0.1` | `bc6355899e1c9462875b6757c79f3caa011a1fdae12bbb1a2e72dd1f234f8336` | `Apache-2.0/MIT` (verbatim upstream Cargo metadata) | `LICENSE-MIT`, `LICENSE-APACHE` |
| `ordered-float` | `5.4.0` | `c860fd3227ca4ac3cc032e2cd20cba3f02ccdf4b610538f8ee6584d56bb62e96` | `MIT` | `LICENSE-MIT` |
| `naga` | `30.0.1` | `a616d2fb8c89516ac2723a581f69d6c18576046bed761bd6b305e5618e6ae130` | `MIT OR Apache-2.0` | `LICENSE.MIT`, `LICENSE.APACHE` |

The inspected bundled license-file SHA-256 values were:

- WGPU 30.0.1 `LICENSE.APACHE`: `a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9`;
- WGPU 30.0.1 `LICENSE.MIT`: `dc0d97139e8205818c703741c7be7cb3b96888bd5917b8d6fc6133731e403c21`;
- Naga 30.0.1 `LICENSE.APACHE`: `a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9`;
- Naga 30.0.1 `LICENSE.MIT`: `dc0d97139e8205818c703741c7be7cb3b96888bd5917b8d6fc6133731e403c21`;
- Pollster 1.0.1 `LICENSE-APACHE`: `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`;
- Pollster 1.0.1 `LICENSE-MIT`: `cc02a53316678ca2575cc69d12daa7b1833935d4b7742c31fbab7bd74a481077`;
- ordered-float 5.4.0 `LICENSE-MIT`: `f7715d38a3fa1b4ac97c5729740752505a39cb92ee83ab5b102aeb5eaa7cdea4`.

This evidence records upstream metadata and bundled license material for the exact locked direct packages only. It does not transfer third-party copyright to Memorithm, replace the upstream grants, establish that all transitive/workspace/fuzz dependencies have been audited, or waive release-time notice/redistribution review. The repository's PolyForm terms apply only to material for which those terms are valid; they do not relicense third-party code.

Any future change to `Cargo.lock`, a direct dependency declaration, or a workspace/fuzz dependency invalidates this exact-version evidence for the changed graph and requires a new audit.
