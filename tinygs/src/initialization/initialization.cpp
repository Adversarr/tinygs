#include "tinygs/initialization/initialization.hpp"
#include "tinygs/initialization/knn.hpp"
#include "tinygs/initialization/random.hpp"
#include <string>

namespace tinygs {

std::unique_ptr<InitializationBase> create_initialization(const std::string& initialization_type) {
  std::string lower_initialization_type = to_lower(initialization_type);
  
  if (lower_initialization_type == "knn") {
    return std::make_unique<KnnInitialization>();
  } else if (lower_initialization_type == "random") {
    return std::make_unique<RandomInitialization>();
  } else {
    throw std::runtime_error("Unknown initialization type: " + initialization_type);
  }
}

} // namespace tinygs