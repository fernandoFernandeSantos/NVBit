#!/bin/bash
set -x
set -e

SAMPLES_DIR=$HOME/NVIDIA_CUDA-11.3_Samples/0_Simple/matrixMul
RECORD_REGS_DIR=$HOME/NVBit/nvbit_release/tools/record_reg_vals

# Build the instrumenter
cd $RECORD_REGS_DIR
make clean
make -j3
cd -

# go to samples
cd $SAMPLES_DIR
make clean
make -j4


eval eval LD_PRELOAD=$RECORD_REGS_DIR/record_reg_vals.so ./matrixMul -wB=64 -wA=64 -hB=64 -hA=64


exit 0
