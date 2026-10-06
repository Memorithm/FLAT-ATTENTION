# Repetition evidence archive

See [the report](../../PVP3D_THOR_REPETITIONS_AND_DELL_2026_10_07.md).
Every campaign and failed setup remains separate. All confirmatory gains are
rejected; do not aggregate surviving samples while discarding failed invocations.

`archive-r1-r7-manifest.json` hashes the initially collected R1–R7 and D1 text
files. `r8/summary.json`, `dell-d2/summary.json` and `r11/summary.json` record the
later user-requested results. `final-restoration-after-r11.txt` records released
locks and restored services. Audit source files use .txt.

R3–R7 compact logs replace only their enormous left/right array lines. Use the
pinned Rust oracle source and mismatch-delta.csv to reconstruct complete actual
and expected vectors; verify little-endian packed hashes in mismatch manifests.
The original logs stay on Thor and their full hashes/byte counts are retained.
R11's native compact output retains exact mismatch count and first eight tuples.
The R8 stale five-process completion marker and R3/R9 setup defects are explained
in the report; original controller/log bytes remain unchanged.
