#pragma once
#include "tinygs/common.hpp"
#include "tinygs/dataset/dataset.hpp"

namespace tinygs {

/**
 * @brief Dataset for loading png images from a folder. With in-memory and 
 *        pinned memory optimization.
 */
class PngFolderDataset final : public DatasetBase {
public:
  explicit PngFolderDataset(const std::string& folder_path, uint32_t height, uint32_t width);
  virtual ~PngFolderDataset();

  virtual Data operator[](size_t index) const override;

private:
  std::string m_folder_path;
  std::vector<std::string> m_image_paths;
  uint32_t m_height, m_width;
  uint32_t m_channels; // TODO: channels must be 4.
  float* m_data;
};

}