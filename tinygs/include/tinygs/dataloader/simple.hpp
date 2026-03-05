#pragma once
#include "tinygs/dataloader/dataloader.hpp"
#include "tinygs/random/pcg32.hpp"

namespace tinygs {

/// @brief Basic dataloader without acceleration
class SimpleDataLoader : public DataLoaderBase {
public:
  explicit SimpleDataLoader(std::shared_ptr<DatasetBase> dataset);

  ~SimpleDataLoader() = default;

  /// @brief Get next batch from dataset
  /// @param stream CUDA stream for data transfer
  GPUBatchInputOutput next(BackendStream stream);
  GPUBatchInputOutput next() override;

  /// Set the parameters for the dataloader.
  void set_params(const json &params) override;

  /// Get the parameters for the dataloader.
  json get_params() const override;

  /// @brief Reset the dataloader and generate a new permutation
  void reset() override;

private:
  /// @brief Generate a new random permutation of dataset indices
  void generate_permutation();

  std::shared_ptr<BackendBuffer> m_gpu_memory;  ///< GPU buffer for data storage
  pcg32 m_rng;                    ///< Random number generator
  std::vector<size_t> m_permutation;  ///< Current permutation of dataset indices
  size_t m_current_index;         ///< Current position in the permutation
};

}  // namespace tinygs
