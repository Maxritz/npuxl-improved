#!/bin/sh
set -eu

cd "$(dirname "$0")"

if [ ! -f test_matmul_batched ]; then
  echo "build test binary first" >&2
  exit 1
fi

SHAPE="128x128x128"
RUN=3
FAIL=0

for run in $(seq 1 $RUN); do
  echo "=== run $run : $SHAPE CPU vs Voodoo vs Wildcat ==="
  if ! ./test_matmul_batched | sed -n '/128x128x128/,$p'; then
    FAIL=1
  fi
done

if [ "$FAIL" = "0" ]; then
  echo "OK: all crosschecks passed"
else
  echo "FAILED: a crosscheck mismatch or run failure occurred"
  exit 1
fi
