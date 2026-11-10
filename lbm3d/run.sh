#!/bin/bash
# Usage: ./lbm3d/run.sh <case in lbm3d/data, without .txt> <frames> [lbm3d options, e.g. -gpu -threads=8]
# Runs the simulation, then plots the mid plane slices of every frame (needs numpy + matplotlib).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "$#" -lt 2 ]; then
    "$HERE/../build3d/lbm3d"
    exit 1
fi
CASE="$1"
FRAMES="$2"
shift 2
mkdir -p "$HERE/outputs3d"
rm -f "$HERE"/outputs3d/frame_*.vtk "$HERE"/outputs3d/frame_*.png
"$HERE/../build3d/lbm3d" "$HERE/data/$CASE.txt" "$FRAMES" -out="$HERE/outputs3d" "$@"
if [ "$FRAMES" -gt 0 ]; then
    python3 "$HERE/scripts/plotting3D.py" "$HERE/outputs3d" -gif
fi
