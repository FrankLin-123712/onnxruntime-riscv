#include "utils.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "stb_image.h"

namespace dpvo {
namespace {

static bool HasSuffixCI(const std::string& str, const std::string& suffix) {
  if (str.size() < suffix.size()) return false;
  size_t start = str.size() - suffix.size();
  for (size_t i = 0; i < suffix.size(); ++i) {
    unsigned char a = static_cast<unsigned char>(str[start + i]);
    unsigned char b = static_cast<unsigned char>(suffix[i]);
    if (std::tolower(a) != std::tolower(b)) return false;
  }
  return true;
}

static bool IsImageFile(const std::string& name) {
  return HasSuffixCI(name, ".png") || HasSuffixCI(name, ".jpg") || HasSuffixCI(name, ".jpeg");
}

static std::string Trim(const std::string& s) {
  size_t start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

static std::vector<std::string> LoadImageListFile(const std::string& path) {
  std::vector<std::string> files;
  std::ifstream f(path);
  if (!f) {
    std::cerr << "ListImages: failed to open list file \"" << path << "\": "
              << std::strerror(errno) << "\n";
    return files;
  }
  std::cout << "Loading image list from \"" << path << "\"\n";
  std::string line;
  while (std::getline(f, line)) {
    line = Trim(line);
    if (line.empty() || line[0] == '#') continue;
    files.push_back(line);
  }
  std::sort(files.begin(), files.end());
  return files;
}

}  // namespace

bool OrtDebugEnabled() {
  const char* v = std::getenv("DPVO_ORT_DEBUG");
  return v && v[0] != '\0';
}

const char* OrtTypeToString(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "double";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "uint32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "uint64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
    default: return "unknown";
  }
}

void PrintShape(std::ostream& os, const std::vector<int64_t>& shape) {
  os << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) os << ",";
    if (shape[i] < 0) os << "?";
    else os << shape[i];
  }
  os << "]";
}

void DumpSessionIO(Ort::Session& sess,
                   const std::vector<const char*>& in_names,
                   const std::vector<const char*>& out_names,
                   const char* tag) {
  try {
    std::cerr << "[dpvo_runner] " << tag << " inputs:\n";
    size_t n_in = sess.GetInputCount();
    for (size_t i = 0; i < n_in; ++i) {
      const char* name = (i < in_names.size()) ? in_names[i] : nullptr;
      std::cerr << "  [" << i << "] " << (name ? name : "(null)");
      auto type_info = sess.GetInputTypeInfo(i);
      if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        std::cerr << " type=" << OrtTypeToString(tensor_info.GetElementType()) << " shape=";
        PrintShape(std::cerr, shape);
      }
      std::cerr << "\n";
    }
    std::cerr << "[dpvo_runner] " << tag << " outputs:\n";
    size_t n_out = sess.GetOutputCount();
    for (size_t i = 0; i < n_out; ++i) {
      const char* name = (i < out_names.size()) ? out_names[i] : nullptr;
      std::cerr << "  [" << i << "] " << (name ? name : "(null)");
      auto type_info = sess.GetOutputTypeInfo(i);
      if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        std::cerr << " type=" << OrtTypeToString(tensor_info.GetElementType()) << " shape=";
        PrintShape(std::cerr, shape);
      }
      std::cerr << "\n";
    }
  } catch (const Ort::Exception& e) {
    std::cerr << "[dpvo_runner] DumpSessionIO failed: " << e.what() << "\n";
  }
}

void DumpRuntimeInputs(const std::vector<const char*>& names,
                       const std::vector<Ort::Value>& inputs,
                       const char* tag) {
  std::cerr << "[dpvo_runner] " << tag << " runtime inputs:\n";
  for (size_t i = 0; i < inputs.size(); ++i) {
    const char* name = (i < names.size()) ? names[i] : nullptr;
    std::cerr << "  [" << i << "] " << (name ? name : "(null)");
    if (!inputs[i].IsTensor()) {
      std::cerr << " (non-tensor)\n";
      continue;
    }
    auto info = inputs[i].GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    std::cerr << " type=" << OrtTypeToString(info.GetElementType()) << " shape=";
    PrintShape(std::cerr, shape);
    std::cerr << "\n";
  }
}

void CopyTensor(const Ort::Value& v, std::vector<float>& out) {
  const float* p = v.GetTensorData<float>();
  auto info = v.GetTensorTypeAndShapeInfo();
  size_t total = info.GetElementCount();
  out.assign(p, p + total);
}

std::vector<float> AvgPool(const std::vector<float>& src,
                           int C,
                           int H,
                           int W,
                           int f,
                           int& Ho,
                           int& Wo) {
  Ho = H / f;
  Wo = W / f;
  std::vector<float> dst(C * Ho * Wo, 0.f);
  for (int c = 0; c < C; ++c) {
    for (int i = 0; i < Ho; ++i) {
      for (int j = 0; j < Wo; ++j) {
        float sum = 0.f;
        for (int di = 0; di < f; ++di) {
          for (int dj = 0; dj < f; ++dj) {
            int si = i * f + di;
            int sj = j * f + dj;
            sum += src[(c * H + si) * W + sj];
          }
        }
        dst[(c * Ho + i) * Wo + j] = sum / (f * f);
      }
    }
  }
  return dst;
}

std::vector<std::string> ListImages(const std::string& dir) {
  if (HasSuffixCI(dir, ".txt")) {
    return LoadImageListFile(dir);
  }
  std::vector<std::string> files;
  DIR* d = opendir(dir.c_str());
  if (!d) {
    std::cerr << "ListImages: failed to open \"" << dir << "\": "
              << std::strerror(errno) << "\n";
    return files;
  }
  std::cout << "directory open successfully.\n";

  while (auto* ent = readdir(d)) {
    std::string name = ent->d_name;
    if (name == "." || name == "..") continue;
    if (IsImageFile(name)) {
      files.push_back(dir + "/" + name);
    }
  }
  closedir(d);
  std::sort(files.begin(), files.end());
  return files;
}

bool LoadImageCHW(const std::string& path, std::vector<float>& out, int64_t& H, int64_t& W) {
  int x, y, c;
  unsigned char* data = stbi_load(path.c_str(), &x, &y, &c, 3);
  if (!data) return false;
  H = y;
  W = x;
  out.resize(3 * H * W);
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      int idx = (i * W + j) * 3;
      int o = i * W + j;
      float r = data[idx + 0] / 255.0f;
      float g = data[idx + 1] / 255.0f;
      float b = data[idx + 2] / 255.0f;
      out[o] = r;
      out[H * W + o] = g;
      out[2 * H * W + o] = b;
    }
  }
  stbi_image_free(data);
  return true;
}

std::vector<float> LoadCalibration(const std::string& path) {
  std::vector<float> k(4, 0.f);
  if (path.empty()) return k;
  std::ifstream f(path);
  for (int i = 0; i < 4 && f; ++i) f >> k[i];
  return k;
}

}  // namespace dpvo
