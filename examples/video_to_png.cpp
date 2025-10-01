#include <cxxopts.hpp>
#include <opencv2/opencv.hpp>

#include "tinygs/dataset/dataset.hpp"
#include "tinygs/dataset/png_folder.hpp"
#include "tinygs/dataset/video.hpp"
#include "tinygs/utils/image_format.hpp"

using namespace tinygs;

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::debug);
  cxxopts::Options options("video_to_png", "Convert video to png folder");
  options.add_options()("f,folder", "Video folder path",
                        cxxopts::value<std::string>()->default_value("/data/yzr/Final/1747834320424/"))(
      "i,id", "Video id", cxxopts::value<uuid_t>()->default_value("1747834320424"))(
      "o,output", "Output folder path", cxxopts::value<std::string>()->default_value("./images"));

  auto result = options.parse(argc, argv);
  std::string folder = result["folder"].as<std::string>();
  uuid_t id = result["id"].as<uuid_t>();
  std::string output = result["output"].as<std::string>();

  log_info("video_to_png, folder: {}, id: {}, output: {}", folder, id, output);

  std::string extrinsics = folder + "/inputs/slam/images.txt";
  // std::string extrinsics = folder + "/inputs/traj_full.txt.bak";
  std::string intrinsic = folder + "/inputs/slam/cameras.txt";

  VideoDataset input(folder + "/" + std::to_string(id) + "_flip.mp4", folder + "/inputs/videoInfo.txt", extrinsics,
                     intrinsic);

  const auto size = input.size();

  for (size_t i = 0; i < size; i++) {
    auto frame = input[i];
    cv::Mat image(frame.image.shape.height, frame.image.shape.width, CV_8UC3);
    // Convert from our CHW+Tiled RGB format to OpenCV HWC BGR
    to_cv2(reinterpret_cast<uint8_t*>(image.data), reinterpret_cast<const uint8_t*>(frame.image.data),
           frame.image.shape);

    cv::imwrite(output + "/" + std::to_string(frame.timestamp) + ".png", image);
  }

  log_info("Done. Processed {} frames.", size);

  // Now, let us try to load the png folder
  PngFolderDataset png_folder(output, extrinsics, intrinsic);

  auto size2 = png_folder.size();
  if (size != size2) {
    log_error("Size mismatch: input size {} != png_folder size {}", size, size2);
    return 1;
  }

  // Variables to track frame comparison
  int identical_frames = 0;
  int total_comparisons = 0;

  for (size_t i = 0; i < size; i++) {
    auto frame = png_folder[i];
    auto ref_frame = input[i];

    // check the content of frame and ref_frame
    if (frame.timestamp != ref_frame.timestamp) {
      log_error("Timestamp mismatch: frame {} (timestamp {}) != ref_frame {} (timestamp {})", frame.frame_idx,
                frame.timestamp, ref_frame.frame_idx, ref_frame.timestamp);
      continue;  // Skip comparison if timestamps don't match
    }

    // Compare image data between PNG folder frame and video reference frame
    total_comparisons++;

    // Both images should have the same shape and format (CHW, UInt8)
    const auto& png_image = frame.image;
    const auto& ref_image = ref_frame.image;

    // Verify image properties match
    if (png_image.shape.width != ref_image.shape.width || png_image.shape.height != ref_image.shape.height
        || png_image.shape.channel != ref_image.shape.channel) {
      log_error("Image shape mismatch for frame {}: PNG {}x{}x{} vs Video {}x{}x{}", frame.frame_idx,
                png_image.shape.width, png_image.shape.height, png_image.shape.channel, ref_image.shape.width,
                ref_image.shape.height, ref_image.shape.channel);
      continue;
    }

    // Validate image data pointers
    if (png_image.data == nullptr || ref_image.data == nullptr) {
      log_error("Null image data pointer for frame {}: PNG data={}, Video data={}", frame.frame_idx,
                png_image.data != nullptr ? "valid" : "null", ref_image.data != nullptr ? "valid" : "null");
      continue;
    }

    // Convert both to HWC BGR for byte-wise comparison
    cv::Mat png_bgr(png_image.shape.height, png_image.shape.width, CV_8UC3);
    cv::Mat vid_bgr(ref_image.shape.height, ref_image.shape.width, CV_8UC3);

    to_cv2(reinterpret_cast<uint8_t*>(png_bgr.data), reinterpret_cast<const uint8_t*>(png_image.data), png_image.shape);
    to_cv2(reinterpret_cast<uint8_t*>(vid_bgr.data), reinterpret_cast<const uint8_t*>(ref_image.data), ref_image.shape);

    const size_t total_bytes
        = static_cast<size_t>(png_image.shape.width) * png_image.shape.height * png_image.shape.channel;
    const uint8_t* a = reinterpret_cast<const uint8_t*>(png_bgr.data);
    const uint8_t* b = reinterpret_cast<const uint8_t*>(vid_bgr.data);

    size_t diff_count = 0;
    for (size_t idx = 0; idx < total_bytes; ++idx) {
      if (a[idx] != b[idx])
        diff_count++;
    }
    bool images_identical = (diff_count == 0);

    if (images_identical) {
      identical_frames++;
    } else {
      log_error("Frame {} images differ in {} bytes out of {} total bytes ({:.2f}%)", frame.frame_idx, diff_count,
                total_bytes, (double)diff_count / total_bytes * 100.0);
    }
  }

  // Report final statistics
  if (total_comparisons > 0) {
    double identical_percentage = (double)identical_frames / total_comparisons * 100.0;
    log_info("Frame comparison complete: {}/{} consecutive frames are identical ({:.2f}%)", identical_frames,
             total_comparisons, identical_percentage);
  } else {
    log_info("No frame comparisons performed (need at least 2 frames)");
  }

  return 0;
}
