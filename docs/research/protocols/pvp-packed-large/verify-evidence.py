"""Strict post-capture analysis; not a GPU executor or performance validator."""

import hashlib
import itertools
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
for item in manifest:
    data = (root / item["path"]).read_bytes()
    assert len(data) == item["bytes"], item["path"]
    assert hashlib.sha256(data).hexdigest() == item["sha256"], item["path"]

source = (root / "source-revision.txt").read_text().strip()
assert source == sys.argv[2], "unexpected execution source"
control = (root / "block-1/control.log").read_text()
assert control.endswith("reservation_released_exit=0\n")
restoration = json.loads((root / "final-restoration-inspection.json").read_text())
assert restoration["controller_exit"] == 0
assert all(restoration[key] is True for key in [
    "guards_removed", "trace_instance_removed", "restore_timer_inactive", "cooperative_lock_available"
])
assert restoration["binary_sha256"] == (root / "binary-sha256.txt").read_text().strip()
for service, policy in restoration["services"].items():
    assert policy["ActiveState"] == "active" and policy["RefuseManualStart"] == "no"
    assert policy["Restart"] == ("no" if service == "rustdesk.service" else "on-failure")
geometries = [(16384, 2048), (65536, 512), (262144, 128), (16384, 129), (65536, 129)]
banks = ["boundary", "sparse4", "dense32"]
phases = ["source", "transformed", "inverse"]
labels = ["A_oracle", "B_oracle", "A_B"]
expected = set(itertools.product(geometries, banks, range(3), phases, labels))
summaries = []
for run in range(1, 11):
    assert re.findall(rf"^run_{run}_exit=(\d+)$", control, re.M) == ["0"], run
    text = (root / f"block-1/run-{run}/packed-large.log").read_text()
    assert f"PVP_LARGE_PROTOCOL,source={source},rounds=3,mapping=explicit,performance_claim=none" in text
    adapters = [line for line in text.splitlines() if line.startswith("PVP_LARGE_ADAPTER,")]
    assert len(adapters) == 1 and 'backend=Vulkan,' in adapters[0]
    assert 'name="NVIDIA Tegra NVIDIA Thor"' in adapters[0]
    seen = set()
    for line in text.splitlines():
        if not line.startswith("PVP_LARGE_COMPARE,"):
            continue
        match = re.fullmatch(
            r"PVP_LARGE_COMPARE,K=(\d+),G=(\d+),bank=(\w+),round=(\d+),"
            r"phase=(\w+),comparison=(\w+),words=(\d+),mismatched_words=0,"
            r"first=\[\],performance_claim=none", line
        )
        assert match, (run, line)
        k, gates, bank, round_id, phase, label, words = match.groups()
        k, gates, round_id, words = map(int, (k, gates, round_id, words))
        assert words == gates * ((k + 127) // 128) * 4
        key = ((k, gates), bank, round_id, phase, label)
        assert key not in seen, (run, key)
        seen.add(key)
    assert seen == expected and len(seen) == 405, run
    complete = [line for line in text.splitlines() if line.startswith("PVP_LARGE_COMPLETE,")]
    assert complete == ["PVP_LARGE_COMPLETE,cases=15,rounds=3,observations=135,comparisons=405,failed_comparisons=0,performance_claim=none"]
    assert "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out;" in text
    summaries.append({"run": run, "comparisons": len(seen), "mismatched_words": 0})

print(json.dumps({
    "schema": "flat.pvp-packed-large-analysis/v1",
    "source": source,
    "manifest_files_verified": len(manifest),
    "processes": summaries,
    "paired_observations": 1350,
    "comparisons": 4050,
    "unknown_observation_runs": sorted({int(x) for x in re.findall(r"diagnostic_observation_incomplete_run_(\d+)", control)}),
    "accepted_performance_rows": 0,
    "performance_claim": "none",
}, indent=2))
