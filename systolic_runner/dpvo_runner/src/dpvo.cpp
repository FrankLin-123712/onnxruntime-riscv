// DPVO C++ runner (Gemmini/ORT)
// Attempts to mirror DPVO/demo.py control flow: patchify -> corr -> update -> BA -> poses.
// Heavy math kernels (altcorr, fastba) are approximated in CPU code so the runner can execute
// end-to-end without custom CUDA kernels. Replace the placeholder math with optimized kernels
// or custom ORT ops to reach parity with the Python implementation.

#include "dpvo.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <systolic/systolic_provider_factory.h>

#include "utils.h"

namespace {

// Simple timer helper
struct ScopedTimer {
  std::chrono::high_resolution_clock::time_point t0;
  bool enabled;
  std::string label;
  ScopedTimer(const std::string& l, bool en)
      : t0(std::chrono::high_resolution_clock::now()), enabled(en), label(l) {}
  ~ScopedTimer() {
    if (!enabled) return;
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cerr << "[time] " << label << ": " << ms << " ms\n";
  }
};

static Ort::SessionOptions MakeSessionOptions(int exec_mode,
                                              int opt_level,
                                              dpvo::ScatterMaxOp& scatter_op,
                                              Ort::CustomOpDomain& custom_domain) {
  Ort::SessionOptions opts;
  // pk/spike does not allow pthread_create; force single-threaded ORT execution.
  opts.SetIntraOpNumThreads(1);
  opts.SetInterOpNumThreads(1);
  opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Systolic(
      opts, /*use_arena=*/1, /*accelerator_mode=*/(char)exec_mode));
  opts.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(opt_level));
  if (dpvo::OrtDebugEnabled()) {
    // Verbose ORT logs (set DPVO_ORT_DEBUG=1 to enable).
    opts.SetLogSeverityLevel(0);
  }
  // Register custom domain dpvo with scatter_max
  custom_domain.Add(&scatter_op);
  opts.Add(custom_domain);
  return opts;
}

}  // namespace

namespace dpvo {

PatchGraph::PatchGraph() {
  poses.resize(BUFFER_SIZE);
  patches.resize(BUFFER_SIZE * PATCHES_PER_FRAME);
  intrinsics.resize(BUFFER_SIZE, {0, 0, 0, 0});
  tstamps.resize(BUFFER_SIZE, 0);
}

FeatureStore::FeatureStore() : frames(BUFFER_SIZE) {}

DPVORunner::DPVORunner(const RunnerConfig& cfg, Ort::Env& env)
    : cfg_(cfg),
      scatter_op_(),
      custom_domain_("dpvo"),
      opts_(MakeSessionOptions(cfg.exec_mode, cfg.opt_level, scatter_op_, custom_domain_)),
      feature_sess_(env, cfg.feature_model.c_str(), opts_),
      update_sess_(env, cfg.update_model.c_str(), opts_),
      alloc_(),
      rng_(1234) {
  feat_input_ = {feature_sess_.GetInputName(0, alloc_)};
  feat_outputs_ = {feature_sess_.GetOutputName(0, alloc_),
                   feature_sess_.GetOutputName(1, alloc_)};
  upd_inputs_.resize(update_sess_.GetInputCount());
  for (size_t i = 0; i < upd_inputs_.size(); ++i) {
    upd_inputs_[i] = update_sess_.GetInputName(i, alloc_);
  }
  upd_outputs_.resize(update_sess_.GetOutputCount());
  for (size_t i = 0; i < upd_outputs_.size(); ++i) {
    upd_outputs_[i] = update_sess_.GetOutputName(i, alloc_);
  }
  DumpIO();
  if (OrtDebugEnabled()) {
    DumpSessionIO(feature_sess_, feat_input_, feat_outputs_, "feature");
    DumpSessionIO(update_sess_, upd_inputs_, upd_outputs_, "update");
  }
}

void DPVORunner::ProcessSequence(const std::vector<std::string>& images,
                                 const std::vector<float>& calib) {
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  for (size_t idx = 0; idx < images.size(); idx += cfg_.stride) {
#ifdef PRINT_INFO
    std::cout << "[INFO] processing image [" << idx << "]\n";
#endif
    if (static_cast<int>(idx) < cfg_.skip) continue;
    int64_t H = 0;
    int64_t W = 0;
    std::vector<float> image_buf;
    if (!LoadImageCHW(images[idx], image_buf, H, W)) {
      std::cerr << "Failed to load image: " << images[idx] << "\n";
      continue;
    }
    std::array<int64_t, 5> shape{1, 1, 3, H, W};  // [B,N,C,H,W]
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, image_buf.data(), image_buf.size(), shape.data(), shape.size());

    FrameBuffers fb;
    {
      ScopedTimer t("feature", cfg_.timeit);
      auto outs = feature_sess_.Run(Ort::RunOptions{nullptr}, feat_input_.data(),
                                    &input_tensor, 1, feat_outputs_.data(),
                                    feat_outputs_.size());
      // outputs: fmap, imap
      CopyTensor(outs[0], fb.fmap);
      CopyTensor(outs[1], fb.imap);
      auto fshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
      fb.Hf = static_cast<int>(fshape[fshape.size() - 2]);
      fb.Wf = static_cast<int>(fshape[fshape.size() - 1]);
      auto ishape = outs[1].GetTensorTypeAndShapeInfo().GetShape();
      fb.Hi = static_cast<int>(ishape[ishape.size() - 2]);
      fb.Wi = static_cast<int>(ishape[ishape.size() - 1]);
    }

    Patchify(image_buf, H, W, fb);
#ifdef PRINT_INFO
    std::cout << "[INFO] Patchify completed @ img:" << idx << "\n";
#endif
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

void DPVORunner::Summary() const {
  std::cout << "dpvo_runner finished frames=" << pg_.n
            << " edges=" << pg_.edges.size()
            << " (CPU corr/BA approximations; replace with optimized kernels for parity)."
            << std::endl;
}

void DPVORunner::DumpIO() {
  std::cerr << "[dpvo_runner] feature inputs:\n";
  for (size_t i = 0; i < feat_input_.size(); ++i) {
    const char* name = feat_input_[i];
    std::cerr << "  [" << i << "] " << (name ? name : "(null)") << "\n";
    if (!name || name[0] == '\0') {
      std::cerr << "  ERROR: feature input name is empty\n";
    }
  }
  std::cerr << "[dpvo_runner] feature outputs:\n";
  for (size_t i = 0; i < feat_outputs_.size(); ++i) {
    const char* name = feat_outputs_[i];
    std::cerr << "  [" << i << "] " << (name ? name : "(null)") << "\n";
    if (!name || name[0] == '\0') {
      std::cerr << "  ERROR: feature output name is empty\n";
    }
  }
  std::cerr << "[dpvo_runner] update inputs:\n";
  for (size_t i = 0; i < upd_inputs_.size(); ++i) {
    const char* name = upd_inputs_[i];
    std::cerr << "  [" << i << "] " << (name ? name : "(null)") << "\n";
    if (!name || name[0] == '\0') {
      std::cerr << "  ERROR: update input name is empty\n";
    }
  }
  std::cerr << "[dpvo_runner] update outputs:\n";
  for (size_t i = 0; i < upd_outputs_.size(); ++i) {
    const char* name = upd_outputs_[i];
    std::cerr << "  [" << i << "] " << (name ? name : "(null)") << "\n";
    if (!name || name[0] == '\0') {
      std::cerr << "  ERROR: update output name is empty\n";
    }
  }
}

void DPVORunner::Patchify(const std::vector<float>& image, int64_t H, int64_t W, FrameBuffers& fb) {
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
    p.rgb = {static_cast<uint8_t>(image[o] * 255.f),
             static_cast<uint8_t>(image[H * W + o] * 255.f),
             static_cast<uint8_t>(image[2 * H * W + o] * 255.f)};
    fb.patches[i] = p;
    pg_.patches[(pg_.n % BUFFER_SIZE) * PATCHES_PER_FRAME + i] = p;
  }
}

// Build forward/backward edges similar to DPVO __edges_forw/back
void DPVORunner::AppendEdges() {
#ifdef PRINT_INFO
  std::cout << "[INFO] Append edges completed.\n";
#endif
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
void DPVORunner::ComputeCorr(const FrameBuffers& fb_i,
                             const FrameBuffers& fb_j,
                             const Patch& p_i,
                             const Patch& p_j,
                             std::vector<float>& corr_out) {
  // Build feature pyramids level1=fmap, level2=avgpool4
  int H1 = fb_i.Hf;
  int W1 = fb_i.Wf;
  int H2i = 0;
  int W2i = 0;
  int H2j = 0;
  int W2j = 0;
  auto fmap2_i = AvgPool(fb_i.fmap, 128, fb_i.Hf, fb_i.Wf, 4, H2i, W2i);
  auto fmap2_j = AvgPool(fb_j.fmap, 128, fb_j.Hf, fb_j.Wf, 4, H2j, W2j);

  corr_out.resize(2 * 49 * PATCH_SIZE * PATCH_SIZE, 0.f);
  int idx = 0;
  auto accumulate = [&](const std::vector<float>& f1,
                        int Hf1,
                        int Wf1,
                        const std::vector<float>& f2,
                        int Hf2,
                        int Wf2,
                        float sx,
                        float sy,
                        float tx,
                        float ty) {
    for (int dy = -CORR_RAD; dy <= CORR_RAD; ++dy) {
      for (int dx = -CORR_RAD; dx <= CORR_RAD; ++dx) {
        float sum = 0.f;
        for (int pi = -PATCH_SIZE / 2; pi <= PATCH_SIZE / 2; ++pi) {
          for (int pj = -PATCH_SIZE / 2; pj <= PATCH_SIZE / 2; ++pj) {
            int xi = std::clamp<int>(static_cast<int>(sx + pj), 0, Wf1 - 1);
            int yi = std::clamp<int>(static_cast<int>(sy + pi), 0, Hf1 - 1);
            int xj = std::clamp<int>(static_cast<int>(tx + dx + pj), 0, Wf2 - 1);
            int yj = std::clamp<int>(static_cast<int>(ty + dy + pi), 0, Hf2 - 1);
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
  accumulate(fmap2_i, H2i, W2i, fmap2_j, H2j, W2j, p_i.x / 4.f, p_i.y / 4.f,
             p_j.x / 4.f, p_j.y / 4.f);
}

void DPVORunner::UpdateStep(const Ort::MemoryInfo& mem_info) {
#ifdef PRINT_INFO
  std::cout << "[INFO] starting update step.\n";
#endif
  if (pg_.edges.empty()) return;
  size_t E = pg_.edges.size();
  const size_t corr_dim = 2 * 49 * PATCH_SIZE * PATCH_SIZE;
  // Build tensors
  std::vector<float> net(E * DIM, 0.f);
  std::vector<float> ctx(E * DIM, 0.f);
  std::vector<float> corr_all(E * corr_dim, 0.f);
  std::vector<int64_t> ii(E);
  std::vector<int64_t> jj(E);
  std::vector<int64_t> kk(E);

#ifdef PRINT_INFO
  std::cout << "[INFO] starting compute correlation.\n";
#endif
  for (size_t e = 0; e < E; ++e) {
    ii[e] = pg_.edges[e].ii;
    jj[e] = pg_.edges[e].jj;
    kk[e] = pg_.edges[e].kk;
    // ctx from imap at patch location
    const FrameBuffers& fb = feat_store_.frames[(pg_.edges[e].ii / PATCHES_PER_FRAME) % BUFFER_SIZE];
    const Patch& p = pg_.patches[pg_.edges[e].ii];
    int xi = std::clamp<int>(static_cast<int>(p.x), 0, fb.Wi - 1);
    int yi = std::clamp<int>(static_cast<int>(p.y), 0, fb.Hi - 1);
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
  std::array<int64_t, 3> net_shape{1, static_cast<int64_t>(E), DIM};
  std::array<int64_t, 3> ctx_shape{1, static_cast<int64_t>(E), DIM};
  std::array<int64_t, 3> corr_shape{1, static_cast<int64_t>(E), static_cast<int64_t>(corr_dim)};
  std::array<int64_t, 3> flow_shape{1, static_cast<int64_t>(E), 2};
  std::array<int64_t, 1> idx_shape{static_cast<int64_t>(E)};
  std::vector<float> flow(E * 2, 0.f);

  Ort::Value net_t = Ort::Value::CreateTensor<float>(
      mem_info, net.data(), net.size(), net_shape.data(), net_shape.size());
  Ort::Value ctx_t = Ort::Value::CreateTensor<float>(
      mem_info, ctx.data(), ctx.size(), ctx_shape.data(), ctx_shape.size());
  Ort::Value corr_t = Ort::Value::CreateTensor<float>(
      mem_info, corr_all.data(), corr_all.size(), corr_shape.data(), corr_shape.size());
  Ort::Value flow_t = Ort::Value::CreateTensor<float>(
      mem_info, flow.data(), flow.size(), flow_shape.data(), flow_shape.size());
  Ort::Value ii_t = Ort::Value::CreateTensor<int64_t>(
      mem_info, ii.data(), ii.size(), idx_shape.data(), idx_shape.size());
  Ort::Value jj_t = Ort::Value::CreateTensor<int64_t>(
      mem_info, jj.data(), jj.size(), idx_shape.data(), idx_shape.size());
  Ort::Value kk_t = Ort::Value::CreateTensor<int64_t>(
      mem_info, kk.data(), kk.size(), idx_shape.data(), idx_shape.size());

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
    if (OrtDebugEnabled()) {
      DumpRuntimeInputs(upd_inputs_, inputs, "update");
    }
    try {
      outputs = update_sess_.Run(Ort::RunOptions{nullptr},
                                 upd_inputs_.data(), inputs.data(), inputs.size(),
                                 upd_outputs_.data(), upd_outputs_.size());
    } catch (const Ort::Exception& e) {
      std::cerr << "ORT exception in update_sess_.Run: " << e.what()
                << " (code=" << e.GetOrtErrorCode() << ")\n";
      DumpRuntimeInputs(upd_inputs_, inputs, "update");
      DumpSessionIO(update_sess_, upd_inputs_, upd_outputs_, "update");
      throw;
    }
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

void DPVORunner::BundleAdjust() {
#ifdef PRINT_INFO
  std::cout << "[INFO] starting bundle adjustment.\n";
#endif
  // Placeholder BA: apply small pose perturbation based on average delta
  if (pg_.edges.empty()) return;
  float mean_dx = 0.f;
  float mean_dy = 0.f;
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

void DPVORunner::KeyframePrune() {
#ifdef PRINT_INFO
  std::cout << "[INFO] starting keyframe pruning.\n";
#endif
  // Simple window pruning to keep edges manageable
  constexpr int WINDOW = 16;
  if (pg_.n <= WINDOW) return;
  int oldest_frame = pg_.n - WINDOW;
  pg_.edges.erase(
      std::remove_if(pg_.edges.begin(), pg_.edges.end(),
                     [&](const Edge& e) { return e.jj < oldest_frame; }),
      pg_.edges.end());
}

}  // namespace dpvo
