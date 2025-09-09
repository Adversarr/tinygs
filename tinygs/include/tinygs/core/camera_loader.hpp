#pragma once

#include <string>
#include <vector>

#include "tinygs/core/camera.hpp"

namespace tinygs {

/**
 * @brief Utility class for loading and storing camera intrinsic and extrinsic parameters from text files.
 *        This class provides methods to read camera parameters in COLMAP format,
 *        making it reusable across different dataset implementations.
 */
class SingleCameraLoader {
public:
  /**
   * @brief Default constructor
   */
  SingleCameraLoader() = default;

  /**
   * @brief Constructor that loads camera parameters from files
   * @param extrinsics_file_path Path to the file containing camera extrinsics
   * @param intrinsics_file_path Path to the file containing camera intrinsics
   */
  SingleCameraLoader(const std::string& extrinsics_file_path, const std::string& intrinsics_file_path);

  /**
   * @brief Load camera extrinsic parameters from a text file
   * @param extrinsics_file_path Path to the file containing camera extrinsics
   * @throws std::runtime_error if file cannot be read or format is invalid
   *
   * Expected format per line: "id qw qx qy qz tx ty tz ..."
   * where:
   * - id: camera frame ID
   * - qw, qx, qy, qz: quaternion components (rotation)
   * - tx, ty, tz: translation components
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

  /**
   * @brief Get the camera extrinsics
   * @return Reference to the vector of camera extrinsics
   */
  const std::vector<CameraExtrinsics>& get_camera_extrinsics() const { return m_camera_extrinsics; }

  /**
   * @brief Get the camera intrinsics
   * @return Reference to the camera intrinsics
   */
  const CameraIntrinsics& get_camera_intrinsics() const { return m_camera_intrinsics; }

private:
  std::vector<CameraExtrinsics> m_camera_extrinsics;
  CameraIntrinsics m_camera_intrinsics;
};

}  // namespace tinygs