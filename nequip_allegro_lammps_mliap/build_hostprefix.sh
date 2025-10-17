#!/usr/bin/env bash
set -euo pipefail

# Build LAMMPS inside stage1.sif into ./hostprefix

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

source "$ROOT_DIR/version.conf"

PREFIX="$ROOT_DIR/hostprefix"
rm -rf "$PREFIX"
mkdir -p "$PREFIX"

echo "Installing to $PREFIX"

PARALLEL=${CMAKE_BUILD_PARALLEL_LEVEL:-8}

cat > build_inside_container.sh <<'CI'
set -e
export PATH=/opt/miniconda3/envs/pytorch/bin:$PATH
ROCM_PATH=/opt/rocm-6.2.4
HIP_PATH=$ROCM_PATH
CMAKE_PREFIX_PATH=$ROCM_PATH
cd /tmp
curl -fsSL -o ${LAMMPS_VERSION}.tar.gz https://github.com/lammps/lammps/archive/refs/tags/${LAMMPS_VERSION}.tar.gz
tar xzf ${LAMMPS_VERSION}.tar.gz
cd lammps-${LAMMPS_VERSION}
mkdir -p build && cd build
cmake ../cmake \
  -D CMAKE_INSTALL_PREFIX="$PREFIX" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER=hipcc \
  -D CMAKE_C_COMPILER=hipcc \
  -D PKG_KOKKOS=ON \
  -D BUILD_MPI=OFF \
  -D BUILD_OMP=ON \
  -D Kokkos_ARCH_ZEN3=ON \
  -D Kokkos_ARCH_VEGA90A=ON \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ENABLE_HIP=ON \
  -D Kokkos_ENABLE_OPENMP=ON \
  -D HIP_PATH=$HIP_PATH \
  -D PKG_ML-IAP=ON \
  -D PKG_ML-SNAP=ON \
  -D MLIAP_ENABLE_PYTHON=ON \
  -D PKG_PYTHON=ON \
  -D BUILD_SHARED_LIBS=ON \
  -D Python_EXECUTABLE=/opt/miniconda3/envs/pytorch/bin/python

make -j${PARALLEL}
make install
make install-python

test -f "$PREFIX/bin/lmp"
test -f "$PREFIX/lib64/liblammps.so" || test -f "$PREFIX/lib/liblammps.so"
CI

chmod +x build_inside_container.sh

echo "Launching build inside container to $PREFIX"
env PREFIX="$PREFIX" LAMMPS_VERSION="$LAMMPS_VERSION" \
  singularity exec "$ROOT_DIR/stage1.sif" bash -lc ./build_inside_container.sh 2>&1 | tee "$ROOT_DIR/build_hostprefix.log"

echo "Done. Binaries in $PREFIX"


