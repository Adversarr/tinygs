#pragma once
#include "tinygs/dataloader/dataloader.hpp"
#include <memory>

namespace tinygs {

/// @brief Asynchronous dataloader, with internal cuda stream.
class AsyncDataLoader : public DataLoaderBase {
public:
  explicit AsyncDataLoader(std::shared_ptr<DatasetBase> dataset);

  /// @brief Destructor
  ~AsyncDataLoader();

  /// @brief Get next batch from dataset
  GPUBatchInputOutput next() override;

  /// @brief Set the parameters for the dataloader
  void set_params(const json &params) override;

  /// @brief Get the parameters for the dataloader
  json get_params() const override;

  /// @brief Reset the dataloader and generate a new permutation
  void reset() override;

private:
  /// @brief Forward declaration of implementation struct
  struct Impl;
  
  /// @brief Pointer to implementation (PIMPL idiom)
  std::unique_ptr<Impl> m_impl;
};

}  // namespace tinygs
