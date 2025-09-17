#include "tinygs/dataset/video.hpp"
#include "tinygs/utils/image_format.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main() {
  using namespace tinygs;
  try {
    VideoDataset video_dataset(
     "/data/accgs/1747834320424/1747834320424_flip.mp4",
     "/data/accgs/1747834320424/inputs/videoInfo.txt",
     "/data/accgs/1747834320424/inputs/traj_full.txt.bak",
     "/data/accgs/1747834320424/inputs/slam/cameras.txt"
   );

   size_t size = video_dataset.size();

   for (size_t i = 0; i < size; i++) {
    auto data = video_dataset[i];

    uint8_t* ptr = (uint8_t*)data.image.data;

    // use opencv to show the image
    cv::Mat image(data.image.shape.height, data.image.shape.width, CV_8UC3);

    chw_to_hwc(ptr, (uint8_t*)image.data, data.image.shape);
    // our image is RGB, but opencv is BGR
    cv::Mat image_bgr;
    cv::cvtColor(image, image_bgr, cv::COLOR_RGB2BGR);

    cv::imshow("Image", image_bgr);
    if (cv::waitKey(0) == 'q') {
      break;
    }
   }

  } catch (const std::exception &e) {
    log_error("Got exception: {}", e.what());
    return -1;
  }
  return 0;
}