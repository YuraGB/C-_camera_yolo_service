#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

struct BBox {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;

    BBox() = default;
    BBox(int x_, int y_, int w_, int h_) : x(x_), y(y_), width(w_), height(h_) {}
    explicit BBox(const cv::Rect& rect)
        : x(rect.x), y(rect.y), width(rect.width), height(rect.height) {}
};

struct Detection {
    std::string label;
    float confidence = 0.0f;
    BBox bbox;

    Detection() = default;
    Detection(std::string label_, float confidence_, const BBox& bbox_)
        : label(std::move(label_)), confidence(confidence_), bbox(bbox_) {}
};

class Frame {
public:
    std::string camera_id;
    int64_t frame_id = 0;
    int64_t timestamp = 0;
    int frame_width = 0;
    int frame_height = 0;

    cv::Mat mat;
    std::vector<Detection> detections;

    Frame() = default;

    Frame(const std::string& cam_id, int64_t id, int64_t ts, const cv::Mat& frame = cv::Mat())
        : camera_id(cam_id),
          frame_id(id),
          timestamp(ts),
          frame_width(frame.cols),
          frame_height(frame.rows),
          mat(frame.empty() ? cv::Mat() : frame.clone()) {}

    int width() const { return mat.empty() ? frame_width : mat.cols; }
    int height() const { return mat.empty() ? frame_height : mat.rows; }
};
