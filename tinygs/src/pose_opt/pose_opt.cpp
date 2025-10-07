#include "tinygs/pose_opt/pose_opt.hpp"
#include "tinygs/pose_opt/none.hpp"
#include "tinygs/pose_opt/sgdm.hpp"
#include "tinygs/pose_opt/adamw.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

std::unique_ptr<PoseOptBase> create_pose_opt(const std::string& pose_opt_type) {
  std::string lower_pose_opt_type = to_lower(pose_opt_type);
  if (lower_pose_opt_type == "none") {
    return std::make_unique<PoseOptNone>();
  } else if (lower_pose_opt_type == "sgdm") {
    return std::make_unique<PoseOptSgdM>();
  } else if (lower_pose_opt_type == "adamw") {
    return std::make_unique<PoseOptAdamW>();
  } else {
    throw std::runtime_error(fmt::format("Unknown pose optimizer type: {}", pose_opt_type));
  }
}

} // namespace tinygs