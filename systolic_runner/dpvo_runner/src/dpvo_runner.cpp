// DPVO C++ runner (Gemmini/ORT)
// Attempts to mirror DPVO/demo.py control flow: patchify -> corr -> update -> BA -> poses.
// Heavy math kernels (altcorr, fastba) are approximated in CPU code so the runner can execute
// end-to-end without custom CUDA kernels. Replace the placeholder math with optimized kernels
// or custom ORT ops to reach parity with the Python implementation.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cxxopts.hpp"
#include "onnxruntime_cxx_api.h"
#include <systolic/systolic_provider_factory.h>
#include "stb_image.h"

namespace {

constexpr int DIM = 384;          // latent dim used by DPVO update block
constexpr int PATCH_SIZE = 3;     // P
constexpr int PATCHES_PER_FRAME = 80;  // M
constexpr int BUFFER_SIZE = 256;  // N (frames)
constexpr int CORR_RAD = 3;       // radius -> 7x7 = 49 locations

struct RunnerConfig {
  std::string feature_model;  // ONNX/TorchScript export of BasicEncoder4 pair
  std::string update_model;   // ONNX export of Update block
  std::string sequence_dir;
  std::string calib_file;
  int stride = 1;
  int skip = 0;
  int exec_mode = 0;    // 0 CPU, 1 OS, 2 WS
  int opt_level = 1;    // ORT graph opt
  bool timeit = false;
};

struct Pose {
  // x,y,z,qx,qy,qz,qw
  std::array<float,7> data{0,0,0,0,0,0,1};
};

struct Patch {
  float x = 0.f, y = 0.f, d = 1.f;   // coords on fmap grid + pseudo-depth
  std::array<uint8_t,3> rgb{{0,0,0}};
};

struct Edge {
  int ii = 0;  // src patch idx
  int jj = 0;  // frame j
  int kk = 0;  // global patch idx
  std::array<float,DIM> net{};   // hidden state
  std::array<float,2> delta{};   // 2D motion delta
  std::array<float,2> weight{{1.f,1.f}};
};

struct FrameBuffers {
  // Feature/imap snapshots for correlation/context
  std::vector<float> fmap;  // [128, Hf, Wf]
  std::vector<float> imap;  // [DIM, Hi, Wi]
  int Hf=0, Wf=0, Hi=0, Wi=0;
  std::vector<Patch> patches;  // size M
};

// Simple timer helper
struct ScopedTimer {
  std::chrono::high_resolution_clock::time_point t0;
  bool enabled;
  std::string label;
  ScopedTimer(const std::string& l, bool en): t0(std::chrono::high_resolution_clock::now()), enabled(en), label(l) {}
  ~ScopedTimer() {
    if (!enabled) return;
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
    std::cerr << "[time] " << label << ": " << ms << " ms\n";
  }
};

// --- IO helpers -------------------------------------------------------------
static std::vector<std::string> ListImages(const std::string& dir) {
  std::vector<std::string> files;
  DIR* d = opendir(dir.c_str());
  if (!d) return files;
  while (auto* ent = readdir(d)) {
    std::string name = ent->d_name;
    if (name == "." || name == "..") continue;
    if (name.find(".png") != std::string::npos || name.find(".jpg") != std::string::npos) {
      files.push_back(dir + "/" + name);
    }
  }
  closedir(d);
  std::sort(files.begin(), files.end());
  return files;
}

static bool LoadImageCHW(const std::string& path, std::vector<float>& out, int64_t& H, int64_t& W) {
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

static std::vector<float> LoadCalibration(const std::string& path) {
  std::vector<float> k(4, 0.f);
  if (path.empty()) return k;
  std::ifstream f(path);
  for (int i = 0; i < 4 && f; ++i) f >> k[i];
  return k;
}

// --- ORT helpers ------------------------------------------------------------
static Ort::Session MakeSession(Ort::Env& env, const std::string& model, int exec_mode, int opt_level) {
  Ort::SessionOptions opts;
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Systolic(opts, /*use_arena=*/1, /*accelerator_mode=*/(char)exec_mode));
  opts.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(opt_level));
  return Ort::Session(env, model.c_str(), opts);
}

// Extract output tensor into std::vector
static void CopyTensor(const Ort::Value& v, std::vector<float>& out) {
  const float* p = v.GetTensorData<float>();
  auto info = v.GetTensorTypeAndShapeInfo();
  size_t total = info.GetElementCount();
  out.assign(p, p + total);
}

// Simple avg pool by factor f along H,W
static std::vector<float> AvgPool(const std::vector<float>& src, int C, int H, int W, int f, int& Ho, int& Wo) {
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

// --- Patch graph + DPVO state ----------------------------------------------
struct PatchGraph {
  int n = 0;          // frames stored
  int m = 0;          // patches stored (n*M)
  std::vector<Pose> poses;               // size BUFFER_SIZE
  std::vector<Patch> patches;            // size BUFFER_SIZE * M
  std::vector<std::array<float,4>> intrinsics;  // fx,fy,cx,cy
  std::vector<int> tstamps;              // time index
  std::vector<Edge> edges;               // active edges
  std::vector<float> target_xy;          // 2 per edge

  PatchGraph() {
    poses.resize(BUFFER_SIZE);
    patches.resize(BUFFER_SIZE * PATCHES_PER_FRAME);
    intrinsics.resize(BUFFER_SIZE, {0,0,0,0});
    tstamps.resize(BUFFER_SIZE, 0);
  }
};

struct FeatureStore {
  // Circular buffers sized by BUFFER_SIZE
  std::vector<FrameBuffers> frames;
  FeatureStore() : frames(BUFFER_SIZE) {}
};

// --- Core runner ------------------------------------------------------------
class DPVORunner {
 public:
  DPVORunner(const RunnerConfig& cfg, Ort::Env& env)
      : cfg_(cfg),
        feature_sess_(MakeSession(env, cfg.feature_model, cfg.exec_mode, cfg.opt_level)),
        update_sess_(MakeSession(env, cfg.update_model, cfg.exec_mode, cfg.opt_level)),
        alloc_() {
    feat_input_ = {feature_sess_.GetInputName(0, alloc_)};
    feat_outputs_ = {feature_sess_.GetOutputName(0, alloc_), feature_sess_.GetOutputName(1, alloc_)};
    upd_inputs_.resize(update_sess_.GetInputCount());
    for (size_t i = 0; i < upd_inputs_.size(); ++i) {
      upd_inputs_[i] = update_sess_.GetInputName(i, alloc_);
    }
    upd_outputs_.resize(update_sess_.GetOutputCount());
    for (size_t i = 0; i < upd_outputs_.size(); ++i) {
      upd_outputs_[i] = update_sess_.GetOutputName(i, alloc_);
    }
  }

  void ProcessSequence(const std::vector<std::string>& images, const std::vector<float>& calib) {
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (size_t idx = 0; idx < images.size(); idx += cfg_.stride) {
      if (static_cast<int>(idx) < cfg_.skip) continue;
      int64_t H = 0, W = 0;
      std::vector<float> image_buf;
      if (!LoadImageCHW(images[idx], image_buf, H, W)) {
        std::cerr << "Failed to load image: " << images[idx] << "\n";
        continue;
      }
      std::array<int64_t, 5> shape{1, 1, 3, H, W};  // [B,N,C,H,W]
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, image_buf.data(),
                                                                image_buf.size(), shape.data(), shape.size());

      FrameBuffers fb;
      {
        ScopedTimer t("feature", cfg_.timeit);
        auto outs = feature_sess_.Run(Ort::RunOptions{nullptr}, feat_input_.data(),
                                      &input_tensor, 1, feat_outputs_.data(), feat_outputs_.size());
        // outputs: fmap, imap
        CopyTensor(outs[0], fb.fmap);
        CopyTensor(outs[1], fb.imap);
        auto fshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
        fb.Hf = static_cast<int>(fshape[fshape.size()-2]);
        fb.Wf = static_cast<int>(fshape[fshape.size()-1]);
        auto ishape = outs[1].GetTensorTypeAndShapeInfo().GetShape();
        fb.Hi = static_cast<int>(ishape[ishape.size()-2]);
        fb.Wi = static_cast<int>(ishape[ishape.size()-1]);
      }

      Patchify(image_buf, H, W, fb);
      feat_store_.frames[pg_.n % BUFFER_SIZE] = fb;

      // record pose prior (motion model: copy prev)
      if (pg_.n > 0) pg_.poses[pg_.n] = pg_.poses[pg_.n - 1];
      pg_.intrinsics[pg_.n] = {calib[0], calib[1], calib[2], calib[3]};
      pg_.tstamps[pg_.n] = static_cast<int>(idx);
      pg_.n++;
      pg_.m = pg_.n * PATCHES_PER_FRAME;

      if (pg_.n > 1) {
        AppendEdges();
        UpdateStep(mem_info);
        BundleAdjust();
        KeyframePrune();
      }
    }
  }

  void Summary() const {
    std::cout << "dpvo_runner finished frames=" << pg_.n
              << " edges=" << pg_.edges.size()
              << " (CPU corr/BA approximations; replace with optimized kernels for parity)." << std::endl;
  }

 private:
  RunnerConfig cfg_;
  Ort::Session feature_sess_;
  Ort::Session update_sess_;
  Ort::AllocatorWithDefaultOptions alloc_;
  std::vector<const char*> feat_input_;
  std::vector<const char*> feat_outputs_;
  std::vector<const char*> upd_inputs_;
  std::vector<const char*> upd_outputs_;

  PatchGraph pg_;
  FeatureStore feat_store_;
  std::mt19937 rng_{1234};

  void Patchify(const std::vector<float>& image, int64_t H, int64_t W, FrameBuffers& fb) {
    // Randomly select patch centers on fmap grid
    std::uniform_int_distribution<int> dx(1, fb.Wf - 2);
    std::uniform_int_distribution<int> dy(1, fb.Hf - 2);
    fb.patches.resize(PATCHES_PER_FRAME);
    for (int i = 0; i < PATCHES_PER_FRAME; ++i) {
      int x = dx(rng_);
      int y = dy(rng_);
      Patch p;
      p.x = static_cast<float>(x);
      p.y = static_cast<float>(y);
      p.d = 1.f;
      // sample color from original image at 4x resolution of fmap
      int ix = std::min<int>(W - 1, x * 4);
      int iy = std::min<int>(H - 1, y * 4);
      size_t o = iy * W + ix;
      p.rgb = { static_cast<uint8_t>(image[o] * 255.f),
                static_cast<uint8_t>(image[H * W + o] * 255.f),
                static_cast<uint8_t>(image[2 * H * W + o] * 255.f) };
      fb.patches[i] = p;
      pg_.patches[(pg_.n % BUFFER_SIZE) * PATCHES_PER_FRAME + i] = p;
    }
  }

  // Build forward/backward edges similar to DPVO __edges_forw/back
  void AppendEdges() {
    int n = pg_.n;
    int m0 = (n - 1) * PATCHES_PER_FRAME;
    int m1 = n * PATCHES_PER_FRAME;
    // forward edges (previous patches -> current frame)
    for (int k = m0; k < m1; ++k) {
      Edge e;
      e.kk = k;
      e.jj = n - 1;
      e.ii = k;  // src patch index
      pg_.edges.push_back(e);
    }
    // backward edges (current patches -> previous frames)
    for (int k = m0; k < m1; ++k) {
      Edge e;
      e.kk = k;
      e.jj = n - 2;
      e.ii = k - PATCHES_PER_FRAME;
      pg_.edges.push_back(e);
    }
  }

  // Correlation volume: naive CPU dot products on two levels
  void ComputeCorr(const FrameBuffers& fb_i, const FrameBuffers& fb_j,
                   const Patch& p_i, const Patch& p_j,
                   std::vector<float>& corr_out) {
    // Build feature pyramids level1=fmap, level2=avgpool4
    int H1 = fb_i.Hf, W1 = fb_i.Wf;
    int H2i, W2i, H2j, W2j;
    auto fmap2_i = AvgPool(fb_i.fmap, 128, fb_i.Hf, fb_i.Wf, 4, H2i, W2i);
    auto fmap2_j = AvgPool(fb_j.fmap, 128, fb_j.Hf, fb_j.Wf, 4, H2j, W2j);

    corr_out.resize(2 * 49 * PATCH_SIZE * PATCH_SIZE, 0.f);
    int idx = 0;
    auto accumulate = [&](const std::vector<float>& f1, int Hf1, int Wf1,
                          const std::vector<float>& f2, int Hf2, int Wf2,
                          float sx, float sy, float tx, float ty) {
      for (int dy = -CORR_RAD; dy <= CORR_RAD; ++dy) {
        for (int dx = -CORR_RAD; dx <= CORR_RAD; ++dx) {
          float sum = 0.f;
          for (int pi = -PATCH_SIZE/2; pi <= PATCH_SIZE/2; ++pi) {
            for (int pj = -PATCH_SIZE/2; pj <= PATCH_SIZE/2; ++pj) {
              int xi = std::clamp<int>(static_cast<int>(sx + pj), 0, Wf1 -1);
              int yi = std::clamp<int>(static_cast<int>(sy + pi), 0, Hf1 -1);
              int xj = std::clamp<int>(static_cast<int>(tx + dx + pj), 0, Wf2 -1);
              int yj = std::clamp<int>(static_cast<int>(ty + dy + pi), 0, Hf2 -1);
              for (int c = 0; c < 128; ++c) {
                float a = f1[(c * Hf1 + yi) * Wf1 + xi];
                float b = f2[(c * Hf2 + yj) * Wf2 + xj];
                sum += a * b;
              }
            }
          }
          corr_out[idx++] = sum;
        }
      }
    };

    accumulate(fb_i.fmap, H1, W1, fb_j.fmap, fb_j.Hf, fb_j.Wf, p_i.x, p_i.y, p_j.x, p_j.y);
    accumulate(fmap2_i, H2i, W2i, fmap2_j, H2j, W2j, p_i.x / 4.f, p_i.y / 4.f, p_j.x / 4.f, p_j.y / 4.f);
  }

  void UpdateStep(const Ort::MemoryInfo& mem_info) {
    if (pg_.edges.empty()) return;
    size_t E = pg_.edges.size();
    const size_t corr_dim = 2 * 49 * PATCH_SIZE * PATCH_SIZE;
    // Build tensors
    std::vector<float> net(E * DIM, 0.f);
    std::vector<float> ctx(E * DIM, 0.f);
    std::vector<float> corr_all(E * corr_dim, 0.f);
    std::vector<int64_t> ii(E), jj(E), kk(E);
    for (size_t e = 0; e < E; ++e) {
      ii[e] = pg_.edges[e].ii;
      jj[e] = pg_.edges[e].jj;
      kk[e] = pg_.edges[e].kk;
      // ctx from imap at patch location
      const FrameBuffers& fb = feat_store_.frames[(pg_.edges[e].ii / PATCHES_PER_FRAME) % BUFFER_SIZE];
      const Patch& p = pg_.patches[pg_.edges[e].ii];
      int xi = std::clamp<int>(static_cast<int>(p.x), 0, fb.Wi -1);
      int yi = std::clamp<int>(static_cast<int>(p.y), 0, fb.Hi -1);
      for (int c = 0; c < DIM; ++c) {
        ctx[e * DIM + c] = fb.imap[(c * fb.Hi + yi) * fb.Wi + xi];
        net[e * DIM + c] = pg_.edges[e].net[c];
      }
      // correlation volume
      const FrameBuffers& fb_j = feat_store_.frames[(pg_.edges[e].jj) % BUFFER_SIZE];
      int local_j = pg_.edges[e].kk % PATCHES_PER_FRAME;
      const Patch& p_j = fb_j.patches[local_j];
      std::vector<float> corr_tmp;
      ComputeCorr(fb, fb_j, p, p_j, corr_tmp);
      std::copy(corr_tmp.begin(), corr_tmp.end(), corr_all.begin() + e * corr_dim);
    }

    // Prepare ORT inputs
    std::array<int64_t,3> net_shape{1, static_cast<int64_t>(E), DIM};
    std::array<int64_t,3> ctx_shape{1, static_cast<int64_t>(E), DIM};
    std::array<int64_t,3> corr_shape{1, static_cast<int64_t>(E), static_cast<int64_t>(corr_dim)};
    std::array<int64_t,3> flow_shape{1, static_cast<int64_t>(E), 2};
    std::array<int64_t,1> idx_shape{static_cast<int64_t>(E)};
    std::vector<float> flow(E * 2, 0.f);

    Ort::Value net_t = Ort::Value::CreateTensor<float>(mem_info, net.data(), net.size(), net_shape.data(), net_shape.size());
    Ort::Value ctx_t = Ort::Value::CreateTensor<float>(mem_info, ctx.data(), ctx.size(), ctx_shape.data(), ctx_shape.size());
    Ort::Value corr_t = Ort::Value::CreateTensor<float>(mem_info, corr_all.data(), corr_all.size(), corr_shape.data(), corr_shape.size());
    Ort::Value flow_t = Ort::Value::CreateTensor<float>(mem_info, flow.data(), flow.size(), flow_shape.data(), flow_shape.size());
    Ort::Value ii_t = Ort::Value::CreateTensor<int64_t>(mem_info, ii.data(), ii.size(), idx_shape.data(), idx_shape.size());
    Ort::Value jj_t = Ort::Value::CreateTensor<int64_t>(mem_info, jj.data(), jj.size(), idx_shape.data(), idx_shape.size());
    Ort::Value kk_t = Ort::Value::CreateTensor<int64_t>(mem_info, kk.data(), kk.size(), idx_shape.data(), idx_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(net_t));
    inputs.push_back(std::move(ctx_t));
    inputs.push_back(std::move(corr_t));
    inputs.push_back(std::move(flow_t));
    inputs.push_back(std::move(ii_t));
    inputs.push_back(std::move(jj_t));
    inputs.push_back(std::move(kk_t));

    std::vector<Ort::Value> outputs;
    {
      ScopedTimer t("update", cfg_.timeit);
      outputs = update_sess_.Run(Ort::RunOptions{nullptr},
                                 upd_inputs_.data(), inputs.data(), inputs.size(),
                                 upd_outputs_.data(), upd_outputs_.size());
    }

    // Extract outputs: assume net_out, delta_weight_misc (flattened)
    auto net_out = outputs[0].GetTensorMutableData<float>();
    auto dw = outputs.size() > 1 ? outputs[1].GetTensorMutableData<float>() : nullptr;
    for (size_t e = 0; e < E; ++e) {
      for (int c = 0; c < DIM; ++c) {
        pg_.edges[e].net[c] = net_out[e * DIM + c];
      }
      if (dw) {
        pg_.edges[e].delta[0] = dw[e * 4 + 0];
        pg_.edges[e].delta[1] = dw[e * 4 + 1];
        pg_.edges[e].weight[0] = dw[e * 4 + 2];
        pg_.edges[e].weight[1] = dw[e * 4 + 3];
      } else {
        pg_.edges[e].delta = {0.f, 0.f};
        pg_.edges[e].weight = {1.f, 1.f};
      }
    }
  }

  void BundleAdjust() {
    // Placeholder BA: apply small pose perturbation based on average delta
    if (pg_.edges.empty()) return;
    float mean_dx = 0.f, mean_dy = 0.f;
    for (const auto& e : pg_.edges) {
      mean_dx += e.delta[0];
      mean_dy += e.delta[1];
    }
    mean_dx /= pg_.edges.size();
    mean_dy /= pg_.edges.size();
    // Translate last pose
    Pose& P = pg_.poses[pg_.n - 1];
    P.data[0] += mean_dx * 0.001f;
    P.data[1] += mean_dy * 0.001f;
  }

  void KeyframePrune() {
    // Simple window pruning to keep edges manageable
    constexpr int WINDOW = 16;
    if (pg_.n <= WINDOW) return;
    int oldest_frame = pg_.n - WINDOW;
    pg_.edges.erase(std::remove_if(pg_.edges.begin(), pg_.edges.end(),
                                   [&](const Edge& e){ return e.jj < oldest_frame; }),
                    pg_.edges.end());
  }
};

RunnerConfig ParseArgs(int argc, char* argv[]) {
  RunnerConfig cfg;
  cxxopts::Options options("dpvo_runner", "DPVO runner skeleton for Gemmini/ORT");
  options.add_options()
      ("feature_model", "Feature extractor ONNX path", cxxopts::value<std::string>())
      ("update_model", "Update block ONNX path", cxxopts::value<std::string>())
      ("sequence_dir", "Directory of frames", cxxopts::value<std::string>())
      ("calib", "Calibration file (fx fy cx cy)", cxxopts::value<std::string>()->default_value(""))
      ("stride", "Frame stride", cxxopts::value<int>()->default_value("1"))
      ("skip", "Frames to skip at start", cxxopts::value<int>()->default_value("0"))
      ("x,execution", "Systolic exec mode (0 CPU,1 OS,2 WS)", cxxopts::value<int>()->default_value("0"))
      ("O,optimization_level", "ORT optimization level", cxxopts::value<int>()->default_value("1"))
      ("timeit", "Print timing info", cxxopts::value<bool>()->default_value("false"))
      ("h,help", "Show help");

  auto res = options.parse(argc, argv);
  if (res.count("help")) {
    std::cout << options.help() << std::endl;
    std::exit(0);
  }
  cfg.feature_model = res["feature_model"].as<std::string>();
  cfg.update_model = res["update_model"].as<std::string>();
  cfg.sequence_dir = res["sequence_dir"].as<std::string>();
  cfg.calib_file = res["calib"].as<std::string>();
  cfg.stride = res["stride"].as<int>();
  cfg.skip = res["skip"].as<int>();
  cfg.exec_mode = res["execution"].as<int>();
  cfg.opt_level = res["optimization_level"].as<int>();
  cfg.timeit = res["timeit"].as<bool>();
  if (cfg.feature_model.empty() || cfg.update_model.empty() || cfg.sequence_dir.empty()) {
    std::cerr << "feature_model, update_model, and sequence_dir are required.\n";
    std::exit(1);
  }
  return cfg;
}

}  // namespace

int main(int argc, char* argv[]) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dpvo");
  RunnerConfig cfg = ParseArgs(argc, argv);

  auto images = ListImages(cfg.sequence_dir);
  if (images.empty()) {
    std::cerr << "No images found in " << cfg.sequence_dir << "\n";
    return 1;
  }
  auto calib = LoadCalibration(cfg.calib_file);

  DPVORunner runner(cfg, env);
  runner.ProcessSequence(images, calib);
  runner.Summary();
  return 0;
}
