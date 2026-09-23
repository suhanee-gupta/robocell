#!/usr/bin/env bash
# End-to-end check: run the CLI headless for several seeds and assert on the JSON summary.
# Exits non-zero on the first failed assertion.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/default.toml}"
TASKS="${TASKS:-500}"
SEEDS=(1 2 3)
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
export MPLBACKEND=Agg

run() { uv run --quiet python -m robocell run --config "$CONFIG" --tasks "$TASKS" --seed "$1" "${@:2}"; }

for seed in "${SEEDS[@]}"; do
    echo "== seed $seed"
    run "$seed" > "$OUT/run_$seed.json"
    uv run --quiet python - "$OUT/run_$seed.json" "$TASKS" "$seed" <<'PY'
import json, sys

path, tasks, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
s = json.load(open(path))
failures = []

def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)

check(s["seed"] == seed, f"seed mismatch: {s['seed']}")
check(s["tasks_generated"] == tasks, f"generated {s['tasks_generated']} != {tasks}")
check(s["finished"] is True, "run did not finish (deadlock or timeout)")
check(
    s["tasks_completed"] + s["tasks_rejected"] == s["tasks_generated"],
    f"completed {s['tasks_completed']} + rejected {s['tasks_rejected']} != generated",
)
check(s["tasks_completed"] > 0, "no task completed")
check(s["limit_violations"] == 0, f"joint limit violations: {s['limit_violations']}")
check(s["max_target_error_m"] < 1e-6, f"target error {s['max_target_error_m']}")
check(set(s["arm_utilization"]) == {"a", "b", "c", "d"}, "missing arm utilisation")
check(all(0.0 < u <= 1.0 for u in s["arm_utilization"].values()), "bad utilisation")
check(0.0 <= s["wait_time_mean_s"] <= s["wait_time_p95_s"] or s["tasks_completed"] < 20,
      "wait time stats inconsistent")

if failures:
    for f in failures:
        print("FAIL:", f)
    sys.exit(1)
print("ok:", json.dumps({k: s[k] for k in ("tasks_completed", "tasks_rejected", "sim_time_s")}))
PY
done

echo "== determinism (seed 1 again)"
run 1 > "$OUT/run_1_again.json"
cmp -s "$OUT/run_1.json" "$OUT/run_1_again.json" || { echo "FAIL: seed 1 output differs between runs"; exit 1; }
echo "ok: identical output"

echo "verify: all checks passed"
