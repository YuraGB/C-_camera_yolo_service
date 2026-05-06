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

    cv::Mat mat;
    std::vector<Detection> detections;
    std::vector<unsigned char> jpeg;

    Frame() = default;

    Frame(const std::string& cam_id, int64_t id, int64_t ts, const cv::Mat& frame = cv::Mat())
        : camera_id(cam_id),
          frame_id(id),
          timestamp(ts),
          mat(frame.empty() ? cv::Mat() : frame.clone()) {}

    int width() const { return mat.cols; }
    int height() const { return mat.rows; }
    int channels() const { return mat.channels(); }

    void encodeJPEG(int quality = 90) {
        if (mat.empty()) {
            return;
        }

        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
        cv::imencode(".jpg", mat, jpeg, params);
    }
};
