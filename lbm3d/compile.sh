#!/bin/bash
# Build lbm3d in release mode into ../build3d. Extra arguments go to cmake, e.g.
#   ./lbm3d/compile.sh -DLBM3D_DOUBLE=ON
#   ./lbm3d/compile.sh -DOpenMP_ROOT=/opt/homebrew/opt/libomp     (macOS, Apple clang + brew libomp)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cmake -S "$HERE" -B "$HERE/../build3d" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$HERE/../build3d" -j
