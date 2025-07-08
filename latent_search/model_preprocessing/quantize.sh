#!/bin/bash
qai-hub submit-compile-job \
  --model           onnx/clip_fp32/clip_full.onnx \
  --device          "Snapdragon X Elite CRD" \
  --device-os       11 \
  --compile_options "
        --target_runtime onnx
  " \
  --input_specs     "{'input_ids': ((1, 77), 'int64'), 'pixel_values': ((1, 3, 224, 224), 'float32'), 'attention_mask': ((1, 77), 'int64')}" \
  --name            clip_full_int8_qdq \
  --wait