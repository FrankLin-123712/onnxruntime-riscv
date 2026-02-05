#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FEAT_MODEL="${SCRIPT_DIR}/onnx_models/feature_extractor_ir7.onnx"
UPD_MODEL="${SCRIPT_DIR}/onnx_models/update_block_ir7.onnx"
SEQ_DIR="${SCRIPT_DIR}/images/subset_0493"
SEQ_LIST="${SCRIPT_DIR}/images/subset_0493.txt"
CALIB="${SCRIPT_DIR}/calib/iphone.txt"
PK=/root/projects/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf/bin/pk

# pk/spike doesn't reliably support directory enumeration; precompute a list.
find "$SEQ_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) \
  | sort > "$SEQ_LIST"

spike --extension=gemmini "$PK" \
  "${SCRIPT_DIR}/dpvo_runner" \
  --feature_model "${FEAT_MODEL}" \
  --update_model "${UPD_MODEL}" \
  --sequence_dir "${SEQ_LIST}" \
  --calib "${CALIB}" \
  --stride 1 --skip 97 \
  -x 0 -O 99
