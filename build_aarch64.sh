#!/bin/bash
set -e



echo "[poky] Sourcing Poky environment..."
source /opt/poky/3.1.31/environment-setup-aarch64-poky-linux

echo "[poky] Cleaning build..."
rm -rf build-poky
mkdir -p build-poky
cd build-poky

echo "[poky] Running CMake..."
# IMPORTANT FIX: Add -DV2H=ON to enable DRP-AI/V2H code
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain/aarch64-toolchain.cmake -DV2H=ON

echo "[poky] Building..."
make -j6

