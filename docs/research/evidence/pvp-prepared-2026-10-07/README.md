# Prepared PVP native evidence

`capture.tar.gz` is a byte-preserving archive of the scoped native campaign:
61 original files, 9940834 uncompressed bytes, plus `manifest.json`.
Archive size: 605730 bytes; SHA256:
`9bceeec99d384a79e4daaeb8cf3eb7d5dfee3d2f6b6d439d4c8dfff2e596a6ab`.
Git blob: `58fcaf1db0f2139bfcb5ad034c02444bdbc335d0`.

`analysis.json` contains strict native verification results; the same verifier
passes the independently transferred local archive. `negative-controls.json`
records six fail-closed controls on temporary copies, never original capture
edits. Unknown occupancy remains unknown in all three processes. No timing or
performance admission is present.

The first restoration inspection's empty stdout remains in the archive; the
lock-availability assertion failed in remote process output. Only inspection
was repeated. The successful final receipt and the subsequent owner-free
snapshot do not identify the transient owner or prove uninterrupted exclusivity.

From the repository root:

```bash
capture_dir=$(mktemp -d /var/tmp/pvp-prepared-verification.XXXXXXXX)
sha256sum docs/research/evidence/pvp-prepared-2026-10-07/capture.tar.gz
tar -xzf docs/research/evidence/pvp-prepared-2026-10-07/capture.tar.gz -C "$capture_dir"
python3 docs/research/protocols/pvp-prepared/verify-evidence.py "$capture_dir" d78599ca0e589a2d578a6523c88441210290c8e1
python3 docs/research/protocols/pvp-prepared/negative-controls.py "$capture_dir" d78599ca0e589a2d578a6523c88441210290c8e1
```

This archive includes service names, process-lifecycle observations and scoped
execution metadata; no credentials or private deployment material are included.
