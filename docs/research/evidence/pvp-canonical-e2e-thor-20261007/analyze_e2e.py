"""Strict finite-corpus integrity and descriptive analysis; never grants admission."""
import itertools
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(sys.argv[1])
SOURCE = "b020d31711d75807eb0f8b13a6a967a8522c5d1c"
GRID = [(4096, 128), (16384, 2048), (65536, 512), (262144, 128), (16384, 129), (65536, 129)]
BANKS = ["boundary", "sparse4", "dense32"]
MODES = ["prepared", "fresh_resources"]
ARMS = ["vec4", "packed7"]
COMPARISONS = ["A_oracle", "B_oracle", "A_B", "canonical_oracle"]
COMPONENTS = ["conversion_in_ns", "preparation_ns", "upload_ns", "compute_wall_ns", "readback_ns", "conversion_out_ns"]


def record(line):
    return dict(part.split("=", 1) for part in line.split(",")[1:])


expected_samples = set(itertools.product(GRID, BANKS, MODES, range(25), ARMS))
expected_checks = set(itertools.product(GRID, BANKS, MODES, range(25), ARMS, COMPARISONS))
runs = []
all_samples = []
for run in range(1, 11):
    lease = ROOT / f"block-{run}"
    log = lease / "run-1/timing.log"
    errors = []
    samples = {}
    checks = set()
    complete = []
    protocols = []
    adapters = []
    rejected = []
    if log.exists():
        for line in log.read_text().splitlines():
            prefix = "test canonical_end_to_end_diagnostic ... "
            if line.startswith(prefix + "PVP_E2E_PROTOCOL,"):
                line = line[len(prefix):]
            if line.startswith("PVP_E2E_PROTOCOL,"):
                protocols.append(record(line))
            elif line.startswith("PVP_E2E_ADAPTER,"):
                adapters.append(record(line))
            elif line.startswith("PVP_E2E_REJECT,"):
                rejected.append(line)
            elif line.startswith("PVP_E2E_COMPLETE,"):
                complete.append(record(line))
            elif line.startswith(("PVP_E2E_SAMPLE,", "PVP_E2E_CHECK,")):
                r = record(line)
                k, g, trial = int(r["K"]), int(r["G"]), int(r["trial"])
                key = ((k, g), r["bank"], r["mode"], trial, r["arm"])
                if line.startswith("PVP_E2E_CHECK,"):
                    check_key = (*key, r["comparison"])
                    if check_key in checks:
                        errors.append("duplicate_check")
                    checks.add(check_key)
                    words = g * ((k + 63) // 64) if r["comparison"] == "canonical_oracle" else (
                        g * ((k + 127) // 128) * 4 + 4 if r["arm"] == "packed7" else k * ((g + 127) // 128) * 4 + 4
                    )
                    if int(r["wrong"]) != 0 or int(r["words"]) != words:
                        errors.append("wrong_check_or_length")
                else:
                    if key in samples:
                        errors.append("duplicate_sample")
                    samples[key] = r
                    measured = trial >= 5
                    repeat = trial - 5 if measured else trial
                    index = ARMS.index(r["arm"])
                    if r["measured"] != str(measured).lower() or int(r["repeat"]) != repeat or int(r["round"]) != trial % 3:
                        errors.append("trial_identity")
                    if int(r["position"]) != (index - repeat % 2) % 2:
                        errors.append("balanced_position")
                    if r["exact"] != "true" or r["admission"] != "none":
                        errors.append("sample_verdict")
                    for field in ["wall_ns", *COMPONENTS]:
                        r[field] = int(r[field])
                        if r[field] <= 0:
                            errors.append("nonpositive_time")
                    if sum(r[f] for f in COMPONENTS) > r["wall_ns"]:
                        errors.append("component_sum_exceeds_wall")
                    r["run"] = run
                    all_samples.append(r)
    else:
        errors.append("missing_timing_log")
    if set(samples) != expected_samples:
        errors.append("sample_coverage")
    if checks != expected_checks:
        errors.append("check_coverage")
    if len(protocols) != 1 or protocols[0].get("source") != SOURCE:
        errors.append("protocol_source")
    if len(adapters) != 1 or adapters[0] != {"name": '"NVIDIA Tegra NVIDIA Thor"', "backend": "Vulkan", "driver": '"580.00"', "type": "IntegratedGpu"}:
        errors.append("adapter_identity")
    if complete != [{"samples": "1440", "rejected_geometries": "0", "failed_comparisons": "0", "performance_admission": "none"}] or rejected:
        errors.append("completion_or_rejection")
    control = (lease / "control.log").read_text() if (lease / "control.log").exists() else ""
    if "run_1_exit=0\n" not in control:
        errors.append("process_exit")
    receipt_path = ROOT / f"restoration-{run}.json"
    if not receipt_path.exists() and run == 7:
        receipt_path = ROOT / "restoration-7-reinspection.stdout"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    restoration = all(receipt.get(k) is True for k in ["guards_removed", "trace_instance_removed", "restore_timer_inactive", "cooperative_lock_available"])
    if not restoration:
        errors.append("restoration")
    occupancy = [json.loads(x) for x in (lease / "occupancy.jsonl").read_text().splitlines()] if (lease / "occupancy.jsonl").exists() else []
    observed = bool(occupancy) and all(x["status"] == "observed" and x["unreadable_entries"] == 0 and not x["gaps"] and not x["scan_limit_reached"] for x in occupancy)
    lifecycle = json.loads((lease / "kernel-lifecycle.json").read_text()) if (lease / "kernel-lifecycle.json").exists() else {}
    trace_loss_free = bool(lifecycle.get("cpu_stats")) and all(s[k] == "0" for s in lifecycle["cpu_stats"].values() for k in ["overrun", "commit overrun", "dropped events"])
    dispatch = (ROOT / f"dispatch-{run}.log").read_text() if (ROOT / f"dispatch-{run}.log").exists() else ""
    first_inspection = ROOT / f"restoration-{run}.stderr"
    runs.append({"run": run, "sample_records": len(samples), "measured_samples": sum(x["measured"] == "true" for x in samples.values()), "check_records": len(checks), "correctness_complete": not errors, "errors": sorted(set(errors)), "occupancy_scans": len(occupancy), "partial_scans": sum(x["status"] == "partial" for x in occupancy), "unreadable_entries": sum(x["unreadable_entries"] for x in occupancy), "incomplete_observation_markers": control.count("observation_incomplete"), "foreign_visible_user_rejections": dispatch.count("admission_rejected=foreign_visible_user"), "unknown_or_foreign_gap_rejections": dispatch.count("admission_rejected=unknown_or_foreign_gap"), "all_scans_observed": observed, "trace_loss_free": trace_loss_free, "restoration_verified": restoration, "first_restoration_inspection_failed": first_inspection.exists() and bool(first_inspection.read_text().strip())})

lookup = {(r["run"], int(r["K"]), int(r["G"]), r["bank"], r["mode"], int(r["repeat"]), r["arm"]): r for r in all_samples if r["measured"] == "true"}
panels = []
for (k, g), bank, mode in itertools.product(GRID, BANKS, MODES):
    medians = []
    components = {arm: {c: [] for c in ["wall_ns", *COMPONENTS]} for arm in ARMS}
    for run in range(1, 11):
        pairs = [(lookup.get((run, k, g, bank, mode, repeat, "vec4")), lookup.get((run, k, g, bank, mode, repeat, "packed7"))) for repeat in range(20)]
        if all(a is not None and b is not None for a, b in pairs):
            ratios = [a["wall_ns"] / b["wall_ns"] for a, b in pairs]
            medians.append({"run": run, "median_paired_ratio": statistics.median(ratios), "minimum_paired_ratio": min(ratios)})
            for a, b in pairs:
                for r in [a, b]:
                    for c in ["wall_ns", *COMPONENTS]:
                        components[r["arm"]][c].append(r[c])
    latency = {}
    for arm in ARMS:
        latency[arm] = {}
        for c, values in components[arm].items():
            if values:
                v = sorted(values)
                latency[arm][c] = {"n": len(v), "median_ns": statistics.median(v), "p95_ns": v[(95 * len(v) + 99) // 100 - 1]}
    panels.append({"K": k, "G": g, "bank": bank, "mode": mode, "process_medians": medians, "descriptive_process_median_min": min((m["median_paired_ratio"] for m in medians), default=None), "all_ten_descriptive_medians_above_1_05": len(medians) == 10 and all(m["median_paired_ratio"] > 1.05 for m in medians), "latency": latency})

result = {"schema": "flat.pvp-canonical-e2e-analysis/v1", "source": SOURCE, "runs": runs, "panels": panels, "complete_correct_processes": sum(r["correctness_complete"] for r in runs), "measured_samples": sum(r["measured"] == "true" for r in all_samples), "independent_admission_review": "missing", "confirmation_p_value": None, "accepted_performance_rows": 0, "performance_admission": "none", "verdict": "diagnostic_only_pending_independent_admission_and_sampling_review"}
(ROOT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ["complete_correct_processes", "measured_samples", "accepted_performance_rows", "verdict"]}))
print(json.dumps(runs))
for panel in panels:
    print(json.dumps({k: panel[k] for k in ["K", "G", "bank", "mode", "descriptive_process_median_min", "all_ten_descriptive_medians_above_1_05"]}))
