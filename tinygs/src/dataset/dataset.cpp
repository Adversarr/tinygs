#include "tinygs/dataset/dataset.hpp"
#include "tinygs/dataset/image.hpp"

namespace tinygs {

DatasetBase::DatasetBase(std::shared_ptr<BackendRuntime> runtime)
  : m_runtime(runtime) {
}

std::unique_ptr<DatasetBase> create_dataset(const std::string& dataset_type,
                                             std::shared_ptr<BackendRuntime> runtime) {
  std::string lower_dataset_type = to_lower(dataset_type);
  if (lower_dataset_type == "image") {
    return std::make_unique<ImageDataset>(runtime);
  } else {
    throw std::runtime_error("Unknown dataset type: " + dataset_type);
  }
}

}  // namespace tinygs
