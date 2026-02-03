#!/bin/bash
spike --extension=gemmini pk \
  dpvo_runner \
    --feature_model /path/to/feature_extractor.onnx \
    --head_model /path/to/heads.onnx \
    --sequence_dir /path/to/frames \
    --calib /path/to/calib.txt \
    --stride 1 --skip 0 \
    -x 0 -O 1
