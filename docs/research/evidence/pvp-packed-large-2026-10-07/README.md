# Lossless original Thor capture

`capture.tar.gz` expands into 103 original runtime files plus `manifest.json`.
Archive SHA256: `d191ca67dbba77626bad824147a548fe0130bb7067e14b6511ac55ddebb42096`.
Every original file is covered by manifest byte length and SHA256; no raw
failure, partial occupancy record or lifecycle snapshot was omitted.
`analysis.json` is a separately reproducible post-capture analysis.

From the repository root:

```sh
capture_dir=$(mktemp -d /var/tmp/pvp-packed-large-verify.XXXXXXXX)
tar -xzf docs/research/evidence/pvp-packed-large-2026-10-07/capture.tar.gz -C "$capture_dir"
python3 docs/research/protocols/pvp-packed-large/verify-evidence.py \
  "$capture_dir" 2065c107f9f32907b21ca944a62aed2ef0d547bd
```

The verifier performs no GPU work or service mutations. The inspector is
read-only but its checks require the original Thor host and lease paths.
The archive excludes the executable itself; its exact pre/post identity,
build messages, compiled/runtime source checks and compiler identity are retained.

No accepted performance samples or gains are contained in this capture.
