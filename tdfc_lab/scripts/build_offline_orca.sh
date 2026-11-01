#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p .offline_build
printf '#define RVO3D_EXPORT\n' > .offline_build/Export.h
c++ -std=c++11 -O3 -shared -fPIC -I.offline_build -Ithird_party/RVO2-3D/src \
 tdfc_lab/research/orca_bridge.cc third_party/RVO2-3D/src/*.cc \
 -o tdfc_lab/research/liborca.so
