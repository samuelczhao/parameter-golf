#!/bin/bash
set -euo pipefail

# boontft Experiment Runner
# Usage: ./run_experiment.sh <experiment_name> [env_overrides...]
# Example: ./run_experiment.sh baseline_ema
# Example: ./run_experiment.sh no_ttt TTT_ENABLED=0
# Example: ./run_experiment.sh bigger_bigram BIGRAM_VOCAB_SIZE=4096

EXPERIMENT="${1:?Usage: $0 <experiment_name> [ENV_VAR=value ...]}"
shift

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate

RESULTS_DIR="boontft/experiments/$(date +%Y-%m-%d)_${EXPERIMENT}"
mkdir -p "$RESULTS_DIR"

SEEDS=(42 1337 2024)
RESULTS_FILE="$RESULTS_DIR/results.json"
echo "[]" > "$RESULTS_FILE"

echo "=== Experiment: $EXPERIMENT ==="
echo "Results: $RESULTS_DIR"
echo "Seeds: ${SEEDS[*]}"
echo "Overrides: $*"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed $SEED ---"
    LOG_FILE="$RESULTS_DIR/seed_${SEED}.log"

    # Apply env overrides
    for override in "$@"; do
        export "$override"
    done

    RUN_ID="${EXPERIMENT}_s${SEED}" \
    SEED="$SEED" \
    TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}" \
    torchrun --standalone --nproc_per_node=1 boontft/train_gpt.py 2>&1 | tee "$LOG_FILE"

    # Extract val_bpb from log
    BPB=$(grep "final_int6_sliding_window_exact" "$LOG_FILE" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    SIZE=$(grep "Total submission size" "$LOG_FILE" | tail -1 | grep -oP '[0-9]+ bytes' | head -1 || echo "N/A")
    TIME=$(grep "train_time:" "$LOG_FILE" | tail -1 | grep -oP 'train_time:\K[0-9]+' || echo "N/A")

    echo "Seed $SEED: val_bpb=$BPB size=$SIZE train_time=${TIME}ms"

    # Append to results
    python3 -c "
import json
with open('$RESULTS_FILE') as f:
    results = json.load(f)
results.append({
    'seed': $SEED,
    'val_bpb': '$BPB',
    'artifact_bytes': '$SIZE',
    'train_time_ms': '$TIME',
    'experiment': '$EXPERIMENT'
})
with open('$RESULTS_FILE', 'w') as f:
    json.dump(results, f, indent=2)
"
done

echo ""
echo "=== Experiment Complete ==="
echo "Results saved to $RESULTS_DIR"

# Summary
python3 -c "
import json, statistics
with open('$RESULTS_FILE') as f:
    results = json.load(f)
bpbs = [float(r['val_bpb']) for r in results if r['val_bpb'] != 'N/A']
if bpbs:
    print(f'Mean val_bpb: {statistics.mean(bpbs):.4f}')
    if len(bpbs) > 1:
        print(f'Std:  {statistics.stdev(bpbs):.5f}')
    print(f'Best: {min(bpbs):.4f}')
"
