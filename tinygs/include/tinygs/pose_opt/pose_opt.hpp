#pragma once
#include "tinygs/cuda/vec.hpp"

namespace tinygs {

class PoseOptBase {
public:
  PoseOptBase() = default;
  virtual ~PoseOptBase() = default;

  /// @brief Query the pose matrix for a given timestamp
  /// @param timestamp Timestamp for which to query the pose matrix
  /// @param world_to_camera World-to-camera pose matrix
  /// @return Pose matrix at the specified timestamp
  virtual mat4x4 query(uuid_t timestamp, mat4x4 world_to_camera) = 0;

  /// @brief Update the pose matrix for a given timestamp
  /// @param timestamp Timestamp for which to update the pose matrix
  /// @param grad_pose Gradient of the pose matrix to be applied at the specified timestamp
  /// @param step_size Step size to scale the gradient update
  /// @throws std::runtime_error if timestamp is not found in the pose matrix
  virtual void update(uuid_t timestamp, const mat4x4& grad_pose, float step_size) = 0;

  /// @brief Get the current optimization parameters as JSON
  /// @return JSON object containing the current parameter values
  virtual json get_params() const noexcept = 0;

  /// @brief Set optimization parameters from JSON
  /// @note The JSON content is expected to match the structure used by get_params()
  virtual void set_params(const json& params) = 0;
};

/// @brief Factory function for creating pose optimizers
/// @param pose_opt_type Type of pose optimizer to create (e.g., "none", "sgdm", "adamw")
/// @return Unique pointer to the created PoseOptBase instance
std::unique_ptr<PoseOptBase> create_pose_opt(const std::string& pose_opt_type);

} // namespace tinygs