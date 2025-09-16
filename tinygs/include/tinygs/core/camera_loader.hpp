#pragma once

#include <string>
#include <vector>

#include "tinygs/core/camera.hpp"

namespace tinygs {

/// @brief Utility class for loading camera parameters from COLMAP format files
class SingleCameraLoader {
public:
  /// @brief Default constructor
  SingleCameraLoader() = default;

  /**
   * @brief Constructor that loads camera parameters from files
   * @param extrinsics_file_path Path to extrinsics file
   * @param intrinsics_file_path Path to intrinsics file
   */
  SingleCameraLoader(const std::string& extrinsics_file_path, const std::string& intrinsics_file_path);

  /**
   * @brief Load camera extrinsic parameters from text file
   * @param extrinsics_file_path Path to extrinsics file
   * @throws std::runtime_error if file cannot be read or format is invalid
   */
  void load_camera_extrinsics(const std::string& extrinsics_file_path);

  /**
   * @brief Load camera intrinsic parameters from a text file
   * @param intrinsics_file_path Path to the file containing camera intrinsics
   * @throws std::runtime_error if file cannot be read, is empty, or format is invalid
   *
   * Expected format: "CAMERA_ID MODEL WIDTH HEIGHT FX FY CX CY K1 K2 K3 P1 P2"
   * where:
   * - CAMERA_ID: camera identifier
   * - MODEL: camera model (e.g., "PINHOLE")
   * - WIDTH, HEIGHT: image dimensions
   * - FX, FY: focal lengths
   * - CX, CY: principal point coordinates
   * - K1, K2, K3: radial distortion coefficients
   * - P1, P2: tangential distortion coefficients
   */
  void load_camera_intrinsics(const std::string& intrinsics_file_path);

  /// @brief Get camera extrinsics sorted by frame_uid
  const std::vector<CameraExtrinsics>& get_camera_extrinsics() const { return m_camera_extrinsics; }

  /// @brief Get camera intrinsics
  const CameraIntrinsics& get_camera_intrinsics() const { return m_camera_intrinsics; }

  void set_camera_intrinsics(const CameraIntrinsics& intrinsics) { m_camera_intrinsics = intrinsics; }

  /**
   * @brief Resize the camera sensor to new image dimensions. The ratio should remain the same.
   * @param width new width
   * @param height new height
   */
  void resize_sensor(uint32_t width, uint32_t height);

private:
  std::vector<CameraExtrinsics> m_camera_extrinsics;
  CameraIntrinsics m_camera_intrinsics;
};

}  // namespace tinygs