"""Strict complete-capture correctness verification; no performance admission."""
import hashlib
import itertools
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
source = sys.argv[2]
manifest = json.loads((root / "manifest.json").read_text())
paths = set()
for item in manifest:
    path = pathlib.PurePosixPath(item["path"])
    assert not path.is_absolute() and ".." not in path.parts
    assert str(path) not in paths, "duplicate manifest file"
    paths.add(str(path))
    data = (root / path).read_bytes()
    assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"], path
assert paths == {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
assert (root / "source-revision.txt").read_text().strip() == source
control = (root / "block-1/control.log").read_text()
assert control.endswith("reservation_released_exit=0\n")
assert "diagnostic_complete=3_invocations_no_timing_claim\n" in control
receipt = json.loads((root / "final-restoration-inspection.json").read_text())
assert receipt["controller_exit"] == 0
assert all(receipt[key] is True for key in ["guards_removed", "trace_instance_removed", "restore_timer_inactive", "cooperative_lock_available"])
assert receipt["binary_sha256"] == (root / "binary-sha256.txt").read_text().strip()
assert receipt["historical_benchmark_sha256"] == "7aef67026ad088504478885cf9f2b7c864218f6fb3b0792bd8609fd907d3f414"
assert set(receipt["services"]) == {"memorithm-clm.service", "memorithm-clm-encoder.service", "memorithm-viggle-lan.service", "rustdesk.service"}
for unit, policy in receipt["services"].items():
    assert policy["ActiveState"] == "active" and policy["RefuseManualStart"] == "no"
    assert policy["Restart"] == ("no" if unit == "rustdesk.service" else "on-failure")

geometries = [(1,1), (2,31), (4,65), (8,129), (16,3), (32,5), (64,257), (128,129),
              (256,513), (512,7), (1024,33), (2048,257), (256,2048),
              (16384,2048), (65536,512), (262144,128)]
arms = ["vec4", "fused2", "tile8", "packed1", "packed7"]
expected = set(itertools.product(arms, geometries, ["boundary", "sparse4", "dense32"], range(3),
                                ["source", "transformed", "inverse"], ["A_oracle", "B_oracle", "A_B"]))
summaries = []
for run in range(1,4):
    assert re.findall(rf"^run_{run}_exit=(\d+)$", control, re.M) == ["0"]
    text = (root / f"block-1/run-{run}/prepared.log").read_text()
    assert text.count(f"PVP_PREPARED_PROTOCOL,source={source},rounds=3,arms=5,geometries=16,mapping=explicit,performance_claim=none") == 1
    adapters = [line for line in text.splitlines() if line.startswith("PVP_PREPARED_ADAPTER,")]
    assert len(adapters) == 1 and "backend=Vulkan," in adapters[0]
    assert 'name="NVIDIA Tegra NVIDIA Thor"' in adapters[0]
    seen = set()
    for line in text.splitlines():
        if not line.startswith("PVP_PREPARED_COMPARE,"):
            continue
        match = re.fullmatch(r"PVP_PREPARED_COMPARE,arm=(\w+),K=(\d+),G=(\d+),bank=(\w+),round=(\d+),"
            r"phase=(\w+),comparison=(\w+),words=(\d+),mismatched_words=0,first=\[\],performance_claim=none", line)
        assert match, (run,line)
        arm,k,g,bank,r,phase,label,words = match.groups()
        k,g,r,words = map(int,(k,g,r,words))
        canonical = g * ((k+127)//128) * 4 if arm.startswith("packed") else k * ((g+127)//128) * 4
        assert words == canonical + 4, (run,line)
        key = (arm,(k,g),bank,r,phase,label)
        assert key not in seen, (run,key)
        seen.add(key)
    assert seen == expected and len(seen) == 6480, run
    assert [line for line in text.splitlines() if line.startswith("PVP_PREPARED_COMPLETE,")] == [
        "PVP_PREPARED_COMPLETE,cases=720,comparisons=6480,failed_comparisons=0,performance_claim=none"]
    assert "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out;" in text
    summaries.append({"run":run,"cases":720,"comparisons":6480,"failed_comparisons":0})

print(json.dumps({"schema":"flat.pvp-prepared-analysis/v1", "source":source,
    "manifest_files_verified":len(manifest), "processes":summaries,
    "paired_observations":6480, "comparisons":19440, "wrong_words":0,
    "unknown_observation_runs":sorted({int(x) for x in re.findall(r"diagnostic_observation_incomplete_run_(\d+)",control)}),
    "accepted_performance_rows":0,"performance_claim":"none"}, indent=2))
