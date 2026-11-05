// Ablation switches for the DPVO Gemmini optimizations.
//
// Every feature is enabled unless its environment variable is exactly "0".
// The defaults therefore keep the deployed (full) behaviour. dpvo_runner sets
// these variables from --opt/--opt_preset before it creates any ORT session;
// each call site reads its switch once per process (function-local static) or
// once per session (graph transformer, capability filter).
//
//   ORT_SYSTOLIC_CONV_RECT         S17  rectangular direct conv on LoopConv
//   ORT_SYSTOLIC_CONV_TILE_SEARCH  S18  capacity-driven direct-conv tile search
//   ORT_SYSTOLIC_NHWC_PROPAGATE    S19  NHWC propagation through CPU Relu/Add/IN
//                                       and Conv/IN/Add -> Relu fusion
//   ORT_SYSTOLIC_FP32_ADD          S20  FP32 Add/AddRelu claimed by the Systolic EP
//   ORT_SYSTOLIC_CONV1X1_NCHW      S21  terminal 1x1 Conv writes NCHW directly
//   ORT_SYSTOLIC_BIAS_RESADD       S22  stride-0 bias resadd (needs S20)
//   ORT_SYSTOLIC_NHWC_NORM         H3   InstanceNormalization_nhwc on Gemmini
#pragma once

#include <cstdlib>
#include <cstring>

inline bool SystolicFeatureEnabled(const char* env_name) {
  const char* value = std::getenv(env_name);
  return value == nullptr || std::strcmp(value, "0") != 0;
}
