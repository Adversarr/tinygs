#include "tinygs/dataset/dataset.hpp"
#include "tinygs/dataset/image.hpp"

namespace tinygs {

std::unique_ptr<DatasetBase> create_dataset(const std::string& dataset_type) {
  std::string lower_dataset_type = to_lower(dataset_type);
  if (lower_dataset_type == "image") {
    return std::make_unique<ImageDataset>();
  } else {
    throw std::runtime_error("Unknown dataset type: " + dataset_type);
  }
}

}  // namespace tinygs
