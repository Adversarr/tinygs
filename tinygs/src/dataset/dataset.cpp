#include "tinygs/dataset/dataset.hpp"
#include "tinygs/dataset/video.hpp"
#include "tinygs/dataset/png_folder.hpp"

namespace tinygs {

std::unique_ptr<DatasetBase> create_dataset(const std::string& dataset_type) {
  std::string lower_dataset_type = to_lower(dataset_type);
  if (lower_dataset_type == "video") {
    return std::make_unique<VideoDataset>();
  } else if (lower_dataset_type == "png_folder") {
    return std::make_unique<PngFolderDataset>();
  } else {
    throw std::runtime_error("Unknown dataset type");
  }
}

}
