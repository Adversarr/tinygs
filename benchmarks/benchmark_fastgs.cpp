#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <memory>
#include <tinygs/core/camera.hpp>
#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/core/pointcloud.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/dataloader/simple.hpp"
#include "tinygs/dataset/png_folder.hpp"
#include "tinygs/initialization/knn.hpp"
#include "tinygs/rasterizer/fastgs.hpp"
#include "tinygs/utils/file.hpp"

std::string DATA_PATH = "/data/accgs/1751090600427/";

struct BenchmarkData {
  std::shared_ptr<tinygs::GPUGaussian3d> gs3d;
  tinygs::FastGSRasterizer rasterizer;
  tinygs::RasterizeContext params;
  tinygs::GPUMemory<float> out_image;
  tinygs::GPUMemory<float> out_alpha;
  tinygs::SimpleDataLoader loader;
  
  BenchmarkData() : out_image(480 * 640 * 3), out_alpha(480 * 640 * 1), 
                    loader(create_dataset()) {
    setup();
  }
  
private:
  std::shared_ptr<tinygs::PngFolderDataset> create_dataset() {
    std::string camera_intrinsics_path = DATA_PATH + "inputs/slam/cameras.txt";
    std::string camera_extrinsics_path = DATA_PATH + "inputs/traj_full.txt.bak";
    
    return std::make_shared<tinygs::PngFolderDataset>(
        DATA_PATH + "inputs/images_480x640_1", camera_extrinsics_path,
        camera_intrinsics_path);
  }
  
  void setup() {
    // Load and initialize point cloud
    auto pc = tinygs::load_from_colmap(DATA_PATH + "inputs/slam/points3D.txt");
    tinygs::KnnInitialization knn;
    knn.initialize(pc);
    auto init_result = knn.gaussians();
    
    // Setup GPU Gaussians
    gs3d = std::make_shared<tinygs::GPUGaussian3d>();
    gs3d->copy_from_host(init_result);
    
    // Setup render parameters
    int width = 480, height = 640;
    tinygs::GPUBatchInputOutput io;
    io.input.width = width;
    io.input.height = height;
    io.input.near = 0.001f;
    io.input.far = 10000.0f;
    
    io.output.image.shape.width = io.output.alpha.shape.width = width;
    io.output.image.shape.height = io.output.alpha.shape.height = height;
    io.output.image.shape.channel = 3;
    io.output.alpha.shape.channel = 1;
    io.output.image.format = io.output.alpha.format = tinygs::ImageFormat::HWC;
    io.output.image.data = out_image.data();
    io.output.alpha.data = out_alpha.data();
    
    params.inference = true;
    params.fwd_input = io.input;
    params.fwd_output = io.output;
    params.gaussians_grad = gs3d;
    
    // Setup rasterizer
    rasterizer.set_gaussians(gs3d);
  }
};

static BenchmarkData& get_benchmark_data() {
  static BenchmarkData data;
  return data;
}

static void fastgs_fwd(benchmark::State& state) {
  auto& data = get_benchmark_data();
  
  // Get a frame for benchmarking
  auto frame_data = data.loader.next();
  data.params.fwd_input.K = frame_data.input.K;
  data.params.fwd_input.w2c = frame_data.input.w2c;
  
  for (auto _ : state) {
    data.rasterizer.forward(data.params);
    cudaDeviceSynchronize(); // Ensure GPU work is complete
  }
}

static void fastgs_fwd_bwd(benchmark::State& state) {
  auto& data = get_benchmark_data();
  
  // Get a frame for benchmarking
  auto frame_data = data.loader.next();
  data.params.fwd_input.K = frame_data.input.K;
  data.params.fwd_input.w2c = frame_data.input.w2c;
  
  for (auto _ : state) {
    data.rasterizer.forward(data.params);
    
    // Setup gradient output
    data.params.grad_output.image = data.params.fwd_output.image;
    data.params.grad_output.alpha = data.params.fwd_output.alpha;
    
    data.rasterizer.backward(data.params);
    cudaDeviceSynchronize(); // Ensure GPU work is complete
  }
}

BENCHMARK(fastgs_fwd);
BENCHMARK(fastgs_fwd_bwd);