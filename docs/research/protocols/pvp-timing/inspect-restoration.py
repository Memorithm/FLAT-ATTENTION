"""Read-only, independent final restoration receipt for the declared Thor lease."""

import hashlib
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
scope = sys.argv[2]
units = ["memorithm-clm.service", "memorithm-clm-encoder.service", "memorithm-viggle-lan.service", "rustdesk.service"]
services = {}
for unit in units:
    output = subprocess.check_output(["systemctl", "show", unit, "--property=ActiveState,Restart,RefuseManualStart"], text=True)
    record = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    assert record["ActiveState"] == "active", unit
    assert record["RefuseManualStart"] == "no", unit
    assert record["Restart"] == ("no" if unit == "rustdesk.service" else "on-failure"), unit
    assert not pathlib.Path(f"/run/systemd/system/{unit}.d/90-{scope}.conf").exists(), unit
    services[unit] = record
assert not pathlib.Path(f"/sys/kernel/tracing/instances/{scope}").exists()
timer = subprocess.run(["systemctl", "is-active", scope + "-restore.timer"], capture_output=True, text=True)
assert timer.returncode != 0 and timer.stdout.strip() in ["inactive", "unknown"]
assert subprocess.run(["flock", "-n", "/dev/nvidia0", "-c", "true"]).returncode == 0
control = (root / "block-1/control.log").read_text()
assert control.endswith("reservation_released_exit=0\n")
assert "diagnostic_complete=3_invocations_raw_timing_no_performance_admission\n" in control
historical = pathlib.Path("/var/tmp/remoteops-pvp-r8-build.c15g0q5t/source/target/release/examples/pvp3d_three_candidate_bench")
historical_hash = hashlib.sha256(historical.read_bytes()).hexdigest()
assert historical_hash == "7aef67026ad088504478885cf9f2b7c864218f6fb3b0792bd8609fd907d3f414"
binary = pathlib.Path((root / "binary-path.txt").read_text().strip())
binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
assert binary_hash == (root / "binary-sha256.txt").read_text().strip()
print(json.dumps({
    "schema": "flat.pvp-timing-restoration/v1",
    "inspected_at_utc": subprocess.check_output(["date", "-u", "--iso-8601=ns"], text=True).strip(),
    "services": services,
    "guards_removed": True,
    "trace_instance_removed": True,
    "restore_timer_inactive": True,
    "cooperative_lock_available": True,
    "controller_exit": 0,
    "binary_sha256": binary_hash,
    "historical_benchmark_sha256": historical_hash,
    "performance_claim": "none",
}, indent=2))
