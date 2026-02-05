#include "cmd_arg.h"

#include <cstdlib>
#include <iostream>
#include <string>

#include "cxxopts.hpp"

namespace dpvo {

RunnerConfig ParseArgs(int argc, char* argv[]) {
  RunnerConfig cfg;
  cxxopts::Options options("dpvo_runner", "DPVO runner skeleton for Gemmini/ORT");
  options.add_options()
      ("feature_model", "Feature extractor ONNX path", cxxopts::value<std::string>())
      ("update_model", "Update block ONNX path", cxxopts::value<std::string>())
      ("sequence_dir", "Directory of frames", cxxopts::value<std::string>())
      ("calib", "Calibration file (fx fy cx cy)",
       cxxopts::value<std::string>()->default_value(""))
      ("stride", "Frame stride", cxxopts::value<int>()->default_value("1"))
      ("skip", "Frames to skip at start", cxxopts::value<int>()->default_value("0"))
      ("x,execution", "Systolic exec mode (0 CPU,1 OS,2 WS)",
       cxxopts::value<int>()->default_value("0"))
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

}  // namespace dpvo
