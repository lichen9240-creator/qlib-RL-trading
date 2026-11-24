#!/usr/bin/env bash
# Deploy beside activate.sh; this script starts experiments only when invoked.
set -eo pipefail
QLIB_SUITE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$QLIB_SUITE_ROOT/activate.sh"
set -u
export PYTHONPATH="$QLIB_SUITE_ROOT/qlib"
cd "$QLIB_SUITE_ROOT"
exec "$QLIB_SUITE_ROOT/envs/qlib/bin/python" -u "$QLIB_SUITE_ROOT/scripts/full_experiment.py" "$@"
