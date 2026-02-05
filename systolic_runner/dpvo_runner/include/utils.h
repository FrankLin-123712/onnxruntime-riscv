#ifndef DPVO_UTILS_H_
#define DPVO_UTILS_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace dpvo {

bool OrtDebugEnabled();
const char* OrtTypeToString(ONNXTensorElementDataType t);
void PrintShape(std::ostream& os, const std::vector<int64_t>& shape);
void DumpSessionIO(Ort::Session& sess,
                   const std::vector<const char*>& in_names,
                   const std::vector<const char*>& out_names,
                   const char* tag);
void DumpRuntimeInputs(const std::vector<const char*>& names,
                       const std::vector<Ort::Value>& inputs,
                       const char* tag);

void CopyTensor(const Ort::Value& v, std::vector<float>& out);
std::vector<float> AvgPool(const std::vector<float>& src,
                           int C,
                           int H,
                           int W,
                           int f,
                           int& Ho,
                           int& Wo);

std::vector<std::string> ListImages(const std::string& dir);
bool LoadImageCHW(const std::string& path, std::vector<float>& out, int64_t& H, int64_t& W);
std::vector<float> LoadCalibration(const std::string& path);

}  // namespace dpvo

#endif  // DPVO_UTILS_H_
