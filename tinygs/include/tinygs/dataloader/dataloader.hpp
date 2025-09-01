#pragma once
#include "tinygs/common.hpp"
namespace tinygs {

using BatchImageDataType = float;

struct GPUBatchData {
  uint32_t batch_size; // Only 1 is support for now.
  uint32_t height, width, channels;

};

class DataLoaderBase {
public:

};
}
