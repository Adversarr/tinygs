#include <benchmark/benchmark.h>
#include <iostream>
#include <tinygs/core/camera.hpp>
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/dataset/png_folder.hpp"
#include "tinygs/utils/file.hpp"
#include "tinygs/dataloader/simple.hpp"

std::shared_ptr<tinygs::PngFolderDataset>& create_test_dataset() {
  static std::shared_ptr<tinygs::PngFolderDataset> dataset;
  if (dataset == nullptr) {
    std::string data_path = "/data/accgs/1747834320424/";
    std::string camera_intrinsics_path = data_path + "inputs/slam/cameras.txt";
    std::string camera_extrinsics_path = data_path + "inputs/traj_full.txt.bak";
    
    dataset = std::make_shared<tinygs::PngFolderDataset>(
        data_path + "inputs/images_480x640_1", camera_extrinsics_path,
        camera_intrinsics_path);
  }
  return dataset;
}

void benchmark_simple_loader(benchmark::State& state) {
  auto& dataset = create_test_dataset();
  tinygs::SimpleDataLoader loader(dataset);

  // Benchmark the data loading
  for (auto _ : state) {
    auto data = loader.next();
    benchmark::DoNotOptimize(data);
  }
}


BENCHMARK(benchmark_simple_loader);
