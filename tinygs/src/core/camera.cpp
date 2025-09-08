#include <tinygs/core/camera.hpp>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace tinygs {

// Implementation of CameraIntrinsics::parse
CameraIntrinsics CameraIntrinsics::parse(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    if (tokens.size() < 13) {
        throw std::runtime_error("Invalid camera intrinsics format: expected 13 values");
    }
    
    CameraIntrinsics intrinsics;
    intrinsics.id = std::stoi(tokens[0]);
    
    // Parse camera model
    if (tokens[1] == "PINHOLE") {
        intrinsics.model = CameraModel::Pinhole;
    } else {
        throw std::runtime_error("Unsupported camera model: " + tokens[1]);
    }
    
    intrinsics.width = std::stoi(tokens[2]);
    intrinsics.height = std::stoi(tokens[3]);
    intrinsics.fx = std::stof(tokens[4]);
    intrinsics.fy = std::stof(tokens[5]);
    intrinsics.cx = std::stof(tokens[6]);
    intrinsics.cy = std::stof(tokens[7]);
    intrinsics.k1 = std::stof(tokens[8]);
    intrinsics.k2 = std::stof(tokens[9]);
    intrinsics.k3 = std::stof(tokens[10]);
    intrinsics.p1 = std::stof(tokens[11]);
    intrinsics.p2 = std::stof(tokens[12]);
    
    return intrinsics;
}

// Implementation of CameraExtrinsics::parse
CameraExtrinsics CameraExtrinsics::parse(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
        if (tokens.size() >= 8) {
            break;
        }
    }

    if (tokens.size() < 8) {
        throw std::runtime_error("Invalid camera extrinsics format: expected at least 8 values, got " + std::to_string(tokens.size()));
    }

    // Parse camera id
    uint32_t frame_uid = std::stoi(tokens[0]);

    // Parse quaternion (qw, qx, qy, qz) and translation (tx, ty, tz)
    float qw = std::stof(tokens[1]);
    float qx = std::stof(tokens[2]);
    float qy = std::stof(tokens[3]);
    float qz = std::stof(tokens[4]);
    float tx = std::stof(tokens[5]);
    float ty = std::stof(tokens[6]);
    float tz = std::stof(tokens[7]);

    // Normalize quaternion
    float norm = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    qw /= norm; qx /= norm; qy /= norm; qz /= norm;

    return CameraExtrinsics(quat{qw, qx, qy, qz}, vec3{tx, ty, tz}, frame_uid);
}

} // namespace tinygs