#!/bin/sh
# Run a shape x dtype benchmark matrix and write machine-readable results.
set -eu
cd "$(dirname "$0")/.."

ITER=${ITER:-5}
OUT=${OUT:-benchmarks}
mkdir -p "$OUT"

echo "Running GEMM matrix (iterations=$ITER)"
./bin/vnpu-cli bench \
  --shape 64x64x64 \
  --shape 128x128x128 \
  --shape 256x256x256 \
  --shape 512x512x512 \
  --shape 512x1024x512 \
  --shape 1024x1024x1024 \
  --dtype both --iterations "$ITER" \
  --json "$OUT/matrix-gemm.json" \
  --trace "$OUT/matrix-trace.json"

echo "Running autotune for representative prefill/decode shapes"
for shape in 64x64x64 512x512x512 512x1024x512 1024x1024x1024; do
  for dtype in f32 i8; do
    slug=$(echo "$shape" | tr 'x' '_')
    ./bin/vnpu-cli autotune --shape "$shape" --dtype "$dtype" \
      --iterations 3 --json "$OUT/tune_${slug}_${dtype}.json"
  done
done

echo "Results written to $OUT"
