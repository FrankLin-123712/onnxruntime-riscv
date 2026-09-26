#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHIPYARD="$(cd "${DIR}/../../../.." && pwd)"
TOOLCHAIN="${CHIPYARD}/.conda-env/riscv-tools"
export PATH="${TOOLCHAIN}/bin:${CHIPYARD}/.conda-env/bin:${PATH}"
export CC="${TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-gcc"
export CXX="${TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++"
BUILD_DIR="${DIR}/build"
for arg in "$@"; do
  case "$arg" in
    --systolic_fp16) BUILD_DIR="${DIR}/build/fp16-riscv" ;;
    --help|-h)
      echo "Usage: ./build.sh --config=Debug|Release --systolic_fp16|--systolic_fp32 [--for_firesim] [--parallel=N]"
      echo "Default: Debug, FP32, Spike (FOR_FIRESIM=OFF). FP16 CPU code requires Zfh."
      echo "Outputs: FP32 build/<config>; FP16 build/fp16-riscv/<config>."
      echo "Compiler: Chipyard .conda-env/riscv-tools (fixed for RISC-V builds)."
      echo "Other ORT options: python3 tools/ci_build/build.py --help"
      exit 0 ;;
  esac
done
for compiler in "$CC" "$CXX"; do
  [[ -x "$compiler" ]] || { echo "Missing Chipyard compiler: $compiler" >&2; exit 1; }
done
cd "${DIR}"
if [[ ! -x build/protoc/bin/protoc ]]; then
  mkdir -p build/protoc
  curl --fail --location "https://github.com/protocolbuffers/protobuf/releases/download/v3.16.0/protoc-3.16.0-linux-x86_64.zip" --output build/protoc/protoc.zip
  unzip -o build/protoc/protoc.zip -d build/protoc
fi
exec python3 "${DIR}/tools/ci_build/build.py" --riscv --skip_submodule_sync \
  --update --build --build_dir="${BUILD_DIR}" "$@"
