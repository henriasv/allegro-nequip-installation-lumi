#!/usr/bin/env bash
# Source this before running inside container to set paths

ROOT_DIR=$(cd "$(dirname "$BASH_SOURCE")" && pwd)

export LD_LIBRARY_PATH=/opt/miniconda3/envs/pytorch/lib:/opt/rocm-6.2.4/lib:/opt/rocm-6.2.4/lib/llvm/lib:$ROOT_DIR/hostprefix/lib64:$ROOT_DIR/hostprefix/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ROOT_DIR/hostprefix/lib/python3.12/site-packages:$ROOT_DIR/hostprefix/lib64/python3.12/site-packages:$PYTHONPATH

echo "LD_LIBRARY_PATH set."

