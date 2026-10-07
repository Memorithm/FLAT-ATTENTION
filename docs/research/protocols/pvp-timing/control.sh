#!/bin/bash
set -euo pipefail
umask 077
root=$1
block=$2
lease_dir="$root/block-$block"
scope="remoteops-pvp-timing-${root##*.}-b$block"
probe=/var/tmp/remoteops-pvp-gap.m_xj95sy/source/target/debug/remoteops-device-users
source_dir=$(cat "$root/source-dir.txt")
binary=$(cat "$root/binary-path.txt")
expected_binary=$(cat "$root/binary-sha256.txt")
guard="90-$scope.conf"
watchdog="$scope-restore"
trace_dir="/sys/kernel/tracing/instances/$scope"
mkdir -p "$lease_dir/xdg"
printf "%s\n" "$$" > "$lease_dir/controller-pid.txt"
printf "%s\n" "$trace_dir" > "$lease_dir/trace-instance-path.txt"
cd "$source_dir"
test "$(git rev-parse HEAD)" = $(cat "$root/source-revision.txt")
test -z "$(git status --porcelain)"
sha256sum "$probe" > "$lease_dir/observer-identity.txt"
test "$(sha256sum "$probe" | cut -d " " -f1)" = 2750a44e6167875081368cd7df6b7ce582e77df17473ce5685727659035149fd
printf '%s  %s\n' "$expected_binary" "$binary" | sha256sum -c - > "$lease_dir/identity-before.txt"
git rev-parse HEAD > "$lease_dir/source-revision.txt"
exec 9<>/dev/nvidia0
flock -n -x 9
if flock -n /dev/nvidia0 -c true 9>&-; then echo reservation_invalid_lock; exit 1; fi
date -u --iso-8601=ns > "$lease_dir/reserved-at.txt"
systemctl show memorithm-clm.service memorithm-clm-encoder.service memorithm-viggle-lan.service rustdesk.service --property=Id,ActiveState,SubState,RefuseManualStart,Restart,TriggeredBy > "$lease_dir/original-services.txt"
for unit in memorithm-clm.service memorithm-clm-encoder.service memorithm-viggle-lan.service rustdesk.service; do
  test "$(systemctl is-active "$unit")" = active
  test "$(systemctl show "$unit" --property=RefuseManualStart --value)" = no
  expected_restart=on-failure
  if [ "$unit" = rustdesk.service ]; then expected_restart=no; fi
  test "$(systemctl show "$unit" --property=Restart --value)" = "$expected_restart"
  test ! -e "/run/systemd/system/$unit.d/$guard"
done
cleanup() {
  result=$?
  trap - EXIT
  if [ -d "$trace_dir" ]; then
    printf "0\\n" > "$trace_dir/tracing_on"
    python3 "$root/check-users.py" --capture "$lease_dir" "$trace_dir" 9>&- || result=1
    printf "0\\n" > "$trace_dir/events/sched/sched_process_fork/enable"
    printf "0\\n" > "$trace_dir/events/sched/sched_process_exit/enable"
    rmdir "$trace_dir" || result=1
  fi
  /bin/bash "$root/restore.sh" "$root" "$block" || result=1
  systemctl stop "$watchdog.timer" || true
  printf 'reservation_released_exit=%s\n' "$result" >> "$lease_dir/control.log"
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
systemd-run --unit="$watchdog" --on-active=30m /bin/bash "$root/restore.sh" "$root" "$block" 9>&-
for unit in memorithm-clm.service memorithm-clm-encoder.service memorithm-viggle-lan.service rustdesk.service; do
  mkdir -p "/run/systemd/system/$unit.d"
  printf '[Unit]\nRefuseManualStart=yes\n[Service]\nRestart=no\n' > "/run/systemd/system/$unit.d/$guard"
done
systemctl daemon-reload
for unit in memorithm-clm.service memorithm-clm-encoder.service memorithm-viggle-lan.service rustdesk.service; do
  test "$(systemctl show "$unit" --property=RefuseManualStart --value)" = yes
done
systemctl stop rustdesk.service memorithm-clm.service memorithm-viggle-lan.service memorithm-clm-encoder.service
phase=admission
check_users() {
  attempts=1
  if [ "$phase" != measuring ]; then attempts=5; fi
  probe_result=2
  for attempt in $(seq 1 "$attempts"); do
    printf '%s\t%s\t%s\n' "$phase" "$attempt" "$(date -u --iso-8601=ns)" >> "$lease_dir/occupancy-times.tsv"
    "$probe" --details /dev/nvidia0 /dev/dri/renderD128 9>&- > "$lease_dir/occupancy-current.json" && probe_result=0 || probe_result=$?
    cat "$lease_dir/occupancy-current.json" >> "$lease_dir/occupancy.jsonl"
    if [ "$probe_result" != 0 ] && [ "$probe_result" != 2 ]; then return "$probe_result"; fi
    checker_result=0
    python3 "$root/check-users.py" "$lease_dir/occupancy-current.json" "$scope" 9>&- >> "$lease_dir/occupancy-decisions.txt" || checker_result=$?
    if [ "$checker_result" = 0 ]; then return 0; fi
    if [ "$checker_result" != 2 ]; then return "$checker_result"; fi
    sleep 1
  done
  return 2
}
mkdir "$trace_dir"
printf "global\\n" > "$trace_dir/trace_clock"
printf "4096\\n" > "$trace_dir/buffer_size_kb"
printf "1\\n" > "$trace_dir/events/sched/sched_process_fork/enable"
printf "1\\n" > "$trace_dir/events/sched/sched_process_exit/enable"
printf "1\\n" > "$trace_dir/tracing_on"
for sample in $(seq 1 5); do check_users; sleep 1; done
printf 'reservation_admitted=controlled_services_paused_lock_contends_visible_users_owned\n' >> "$lease_dir/control.log"
export XDG_RUNTIME_DIR="$lease_dir/xdg"
export WGPU_BACKEND=vulkan FLAT_REQUIRE_WGPU=1 FLAT_PVP_PREPARED_ROUNDS=3
export FLAT_SOURCE_REVISION=$(cat "$root/source-revision.txt")
for invocation in $(seq 1 3); do
  run_dir="$lease_dir/run-$invocation"
  mkdir -p "$run_dir"
  mode=prepared_resident_wall_diagnostic
  printf '%s\n' "$mode" > "$run_dir/mode.txt"
  date -u --iso-8601=ns > "$run_dir/start-at.txt"
  cat /proc/loadavg > "$run_dir/load-before.txt"
  phase=measuring
  timeout 360s "$binary" --exact "$mode" --ignored --nocapture --test-threads=1 > "$run_dir/timing.log" 2>&1 &
  benchmark_job=$!
  while kill -0 "$benchmark_job" 2>/dev/null; do
    if ! check_users; then
      printf 'diagnostic_observation_incomplete_run_%s\n' "$invocation" >> "$lease_dir/control.log"
    fi
    sleep 1
  done
  test_result=0
  wait "$benchmark_job" || test_result=$?
  printf "run_%s_exit=%s\n" "$invocation" "$test_result" >> "$lease_dir/control.log"
  phase=post_measurement
  if ! check_users; then printf "post_diagnostic_observation_incomplete_run_%s\n" "$invocation" >> "$lease_dir/control.log"; fi
  if grep -Fq 'PVP_TIMING_COMPLETE,' "$run_dir/timing.log"; then
    grep -F 'PVP_TIMING_COMPLETE,' "$run_dir/timing.log" >> "$lease_dir/control.log"
  else
    printf 'run_%s_diagnostic=incomplete\n' "$invocation" >> "$lease_dir/control.log"
  fi
  cat /proc/loadavg > "$run_dir/load-after.txt"
  date -u --iso-8601=ns > "$run_dir/complete-at.txt"
done
sha256sum "$binary" > "$lease_dir/identity-after.txt"
printf '%s  %s\n' "$expected_binary" "$binary" | sha256sum -c -
date -u --iso-8601=ns > "$lease_dir/completed-at.txt"
printf 'diagnostic_complete=3_invocations_no_timing_claim\n' >> "$lease_dir/control.log"

