#!/usr/bin/env bash
# Compile an ONNX model with Apache TVM, auto-tune it, and compare run times.
#
# This is the command sequence from notebooks/tvm_compiler_tuning.ipynb, with
# the model, input shape and target turned into variables so it can be pointed
# at any ONNX file (the notebook itself ran it on the ResNet50-v2 tutorial model).
#
# Usage:
#   scripts/tvm_compile_and_tune.sh [MODEL.onnx] [INPUT_SHAPES] [TARGET]
#
# Defaults compile this project's optimized ONNX digit model for a generic CPU:
#   MODEL         models/digit_recognition_optimized.onnx
#   INPUT_SHAPES  conv1_input:[1,28,28,1]
#   TARGET        llvm
#
# Requires: pip install apache-tvm xgboost   (tvmc ships with apache-tvm)
set -euo pipefail

MODEL="${1:-models/digit_recognition_optimized.onnx}"
INPUT_SHAPES="${2:-conv1_input:[1,28,28,1]}"
TARGET="${3:-llvm}"
OUT_DIR="${OUT_DIR:-build/tvm}"
REPEAT="${REPEAT:-100}"

mkdir -p "$OUT_DIR"
name="$(basename "${MODEL%.onnx}")"
inputs_npz="$OUT_DIR/${name}_inputs.npz"

# 1. Compile without tuning (TVM's default schedules).
tvmc compile --target "$TARGET" --input-shapes "$INPUT_SHAPES" \
  --output "$OUT_DIR/${name}-tvm.tar" "$MODEL"

# 2. Auto-tune: search for faster operator schedules on this machine.
#    This is the slow step (the notebook's ResNet50 run took tens of minutes).
tvmc tune --target "$TARGET" --input-shapes "$INPUT_SHAPES" \
  --output "$OUT_DIR/${name}-autotuner_records.json" "$MODEL"

# 3. Re-compile using the tuning records.
tvmc compile --target "$TARGET" --input-shapes "$INPUT_SHAPES" \
  --tuning-records "$OUT_DIR/${name}-autotuner_records.json" \
  --output "$OUT_DIR/${name}-tvm_autotuned.tar" "$MODEL"

# 4. Build a zero-filled input matching the declared shape, so `tvmc run`
#    has something to feed. Timing does not depend on the pixel values.
python - "$INPUT_SHAPES" "$inputs_npz" <<'PY'
import json, sys
import numpy as np
spec, out = sys.argv[1], sys.argv[2]
name, shape = spec.split(":", 1)
np.savez(out, **{name: np.zeros(json.loads(shape), dtype="float32")})
PY

# 5. Time both builds with the same input (mean/median/max/min/std in ms).
for build in "${name}-tvm.tar" "${name}-tvm_autotuned.tar"; do
  echo "== $build"
  tvmc run --inputs "$inputs_npz" --print-time --repeat "$REPEAT" "$OUT_DIR/$build"
done
