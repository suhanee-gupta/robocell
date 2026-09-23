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

if failures:
    for f in failures:
        print("FAIL:", f)
    sys.exit(1)
print("ok:", json.dumps({k: s[k] for k in ("tasks_generated",)}))
PY
done

echo "== determinism (seed 1 again)"
run 1 > "$OUT/run_1_again.json"
cmp -s "$OUT/run_1.json" "$OUT/run_1_again.json" || { echo "FAIL: seed 1 output differs between runs"; exit 1; }
echo "ok: identical output"

echo "verify: all checks passed"
