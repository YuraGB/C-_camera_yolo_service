    #pragma once
    #include <string>
    #include <vector>
    #include <opencv2/opencv.hpp> // cv::Mat

    // ------------------------------
    // Прямокутник для детекцій
    // ------------------------------
    struct BBox {
        int x = 0;
        int y = 0;
        int width = 0;   // тепер сумісно з protobuf
        int height = 0;  // тепер сумісно з protobuf

        BBox() = default;
        BBox(int x_, int y_, int w_, int h_) : x(x_), y(y_), width(w_), height(h_) {}

        // конструктор з cv::Rect
         BBox(const cv::Rect& r) : x(r.x), y(r.y), width(r.width), height(r.height) {}
    };

    // ------------------------------
    // Детекція одного об'єкта
    // ------------------------------
    struct Detection {
        std::string label;    // мітка класу
        float confidence = 0; // ймовірність
        BBox bbox;            // рамка

        int track_id = -1;

        Detection() = default;
        Detection(const std::string& l, float c, const BBox& b)
            : label(l), confidence(c), bbox(b) {}
        Detection(const std::string& l, float c, const BBox& b, int id)
            : label(l), confidence(c), bbox(b), track_id(id) {}
    };

    // ------------------------------
    // Кадр з камери або відео
    // ------------------------------
    class Frame {
    public:
        std::string camera_id;
        int64_t frame_id = 0;
        int64_t timestamp = 0;

        cv::Mat mat;                       // локальний кадр для обробки
        std::vector<Detection> detections; // результат детекцій
        std::vector<unsigned char> jpeg;   // стиснутий JPEG для gRPC

        std::vector<float> inference_result; // сирі дані з моделі YOLO

        Frame() = default;

        Frame(const std::string& cam_id, int64_t id, int64_t ts, const cv::Mat& m = cv::Mat())
            : camera_id(cam_id), frame_id(id), timestamp(ts), mat(m.empty() ? cv::Mat() : m.clone()) {}

        // --------------------------
        // Допоміжні методи для ONNX
        // --------------------------
        int width() const { return mat.cols; }
        int height() const { return mat.rows; }
        int channels() const { return mat.channels(); }

        // Отримати кадр у форматі CHW (HWC -> CHW), нормалізований
std::vector<float> getDataCHW() const {
    std::vector<float> chw;
    int c = mat.channels();
    int h = mat.rows;
    int w = mat.cols;
    chw.resize(c * h * w);

    if (mat.type() == CV_32FC3) {
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
                for(int ch = 0; ch < c; ch++)
                    chw[ch*h*w + y*w + x] = mat.at<cv::Vec3f>(y,x)[ch];
    } else if (mat.type() == CV_8UC3) {
        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
                for(int ch = 0; ch < c; ch++)
                    chw[ch*h*w + y*w + x] = mat.at<cv::Vec3b>(y,x)[ch] / 255.0f;
    }

    return chw;
}

     void parseDetectionsFromYOLO(float confThreshold = 0.25f, float iouThreshold = 0.5f) {
        detections.clear();

        const int num_classes = 80;
        const int elements_per_det = 84; // 4 bbox + 80 класів
        int num_det = static_cast<int>(inference_result.size() / elements_per_det);
        if (num_det == 0) return;

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        for (int i = 0; i < num_det; ++i) {
            float* ptr = inference_result.data() + i * elements_per_det;

            float x = ptr[0];
            float y = ptr[1];
            float w = ptr[2];
            float h = ptr[3];

            // знайти клас з max confidence
            float max_conf = 0;
            int max_class = -1;
            for (int c = 0; c < num_classes; ++c) {
                float conf = ptr[4 + c];
                if (conf > max_conf) {
                    max_conf = conf;
                    max_class = c;
                }
            }

            if (max_conf > confThreshold) {
                boxes.emplace_back(static_cast<int>(x), static_cast<int>(y),
                                static_cast<int>(w), static_cast<int>(h));
                scores.push_back(max_conf);
                class_ids.push_back(max_class);
            }
        }

        // --- NMS ---
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, confThreshold, iouThreshold, indices);

        for (int idx : indices) {
            detections.emplace_back(
                std::to_string(class_ids[idx]), // або мапа класів
                scores[idx],
                BBox(boxes[idx])
            );
        }

        std::cout << "[DEBUG] detections after NMS: " << detections.size() << std::endl;
    }


        // --------------------------
        // Метод для конвертації Mat -> JPEG
        // --------------------------
        void encodeJPEG(int quality = 90) {
            if (mat.empty()) return;
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
            cv::imencode(".jpg", mat, jpeg, params);
        }
    };

    
