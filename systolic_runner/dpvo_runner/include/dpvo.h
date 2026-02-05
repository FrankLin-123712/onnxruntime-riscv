#ifndef DPVO_H_
#define DPVO_H_

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "custom_op.h"
#include "onnxruntime_cxx_api.h"

namespace dpvo {

constexpr int DIM = 384;              // latent dim used by DPVO update block
constexpr int PATCH_SIZE = 3;         // P
constexpr int PATCHES_PER_FRAME = 80; // M
constexpr int BUFFER_SIZE = 256;      // N (frames)
constexpr int CORR_RAD = 3;           // radius -> 7x7 = 49 locations

struct RunnerConfig {
  std::string feature_model;  // ONNX/TorchScript export of BasicEncoder4 pair
  std::string update_model;   // ONNX export of Update block
  std::string sequence_dir;
  std::string calib_file;
  int stride = 1;
  int skip = 0;
  int exec_mode = 0;  // 0 CPU, 1 OS, 2 WS
  int opt_level = 1;  // ORT graph opt
  bool timeit = true;
};

struct Pose {
  std::array<float, 7> data{0, 0, 0, 0, 0, 0, 1};
};

struct Patch {
  float x = 0.f;
  float y = 0.f;
  float d = 1.f;
  std::array<uint8_t, 3> rgb{{0, 0, 0}};
};

struct Edge {
  int ii = 0;  // src patch idx
  int jj = 0;  // frame j
  int kk = 0;  // global patch idx
  std::array<float, DIM> net{};  // hidden state
  std::array<float, 2> delta{};  // 2D motion delta
  std::array<float, 2> weight{{1.f, 1.f}};
};

struct FrameBuffers {
  std::vector<float> fmap;  // [128, Hf, Wf]
  std::vector<float> imap;  // [DIM, Hi, Wi]
  int Hf = 0;
  int Wf = 0;
  int Hi = 0;
  int Wi = 0;
  std::vector<Patch> patches;  // size M
};

struct PatchGraph {
  int n = 0;  // frames stored
  int m = 0;  // patches stored (n*M)
  std::vector<Pose> poses;  // size BUFFER_SIZE
  std::vector<Patch> patches;  // size BUFFER_SIZE * M
  std::vector<std::array<float, 4>> intrinsics;  // fx,fy,cx,cy
  std::vector<int> tstamps;  // time index
  std::vector<Edge> edges;  // active edges
  std::vector<float> target_xy;  // 2 per edge

  PatchGraph();
};

struct FeatureStore {
  std::vector<FrameBuffers> frames;  // circular buffers sized by BUFFER_SIZE
  FeatureStore();
};

class DPVORunner {
 public:
  DPVORunner(const RunnerConfig& cfg, Ort::Env& env);

  void ProcessSequence(const std::vector<std::string>& images,
                       const std::vector<float>& calib);
  void Summary() const;

 private:
  RunnerConfig cfg_;
  ScatterMaxOp scatter_op_;
  Ort::CustomOpDomain custom_domain_;
  Ort::SessionOptions opts_;
  Ort::Session feature_sess_;
  Ort::Session update_sess_;
  Ort::AllocatorWithDefaultOptions alloc_;
  std::vector<const char*> feat_input_;
  std::vector<const char*> feat_outputs_;
  std::vector<const char*> upd_inputs_;
  std::vector<const char*> upd_outputs_;

  PatchGraph pg_;
  FeatureStore feat_store_;
  std::mt19937 rng_;

  void DumpIO();
  void Patchify(const std::vector<float>& image, int64_t H, int64_t W, FrameBuffers& fb);
  void AppendEdges();
  void ComputeCorr(const FrameBuffers& fb_i, const FrameBuffers& fb_j,
                   const Patch& p_i, const Patch& p_j,
                   std::vector<float>& corr_out);
  void UpdateStep(const Ort::MemoryInfo& mem_info);
  void BundleAdjust();
  void KeyframePrune();
};

}  // namespace dpvo

#endif  // DPVO_H_
