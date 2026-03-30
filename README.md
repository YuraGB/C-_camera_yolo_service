# Camera CV Service

Camera CV Service is a C++ application for capturing frames from cameras or video files, running YOLO inference through ONNX Runtime, and publishing the results over gRPC.

The current project version exposes two independent gRPC streams:

- `StreamLiveFrames`: low-latency JPEG frames without detections
- `StreamDetectionFrames`: JPEG frames published after YOLO inference completes, with detections aligned to the same image

This split lets a client display a smooth live view and a slower detection view in parallel.

## Features

- Multiple input sources at the same time
- USB / built-in cameras
- Video files such as `.mp4` and `.avi`
- Frame capture with per-camera worker threads
- YOLO inference through ONNX Runtime
- gRPC streaming for both live and processed frames
- Latest-frame streaming semantics to keep latency bounded

## gRPC API

The protobuf service currently defines two server-streaming RPCs:

```proto
service DetectionService {
  rpc StreamLiveFrames(google.protobuf.Empty) returns (stream Frame);
  rpc StreamDetectionFrames(google.protobuf.Empty) returns (stream Frame);
}
```

Both streams return the same `Frame` message:

```json
{
  "frameId": 123,
  "timestamp": 1711800000123,
  "cameraId": "camera_0",
  "image": "<JPEG bytes>",
  "detections": [
    {
      "label": "person",
      "confidence": 0.92,
      "bbox": {
        "x": 120,
        "y": 48,
        "width": 210,
        "height": 390
      }
    }
  ]
}
```

Expected stream behavior:

- `StreamLiveFrames`
  - sends frames immediately after capture
  - `detections` is usually empty
  - best option for low-latency display
- `StreamDetectionFrames`
  - sends frames only after inference finishes
  - `detections` belongs to the exact image in the same message
  - lower effective FPS and higher latency than the live stream

## Runtime Flow

1. Camera threads capture frames and keep the newest frame for each source.
2. The main loop publishes the newest frame to `StreamLiveFrames`.
3. The same frame is submitted to the inference engine.
4. When YOLO finishes, the processed frame is published to `StreamDetectionFrames`.

This design is intentionally optimized for UX:

- live viewing stays responsive
- detection overlays stay aligned
- stale queue buildup is reduced by keeping the newest data

## Technology Stack

- C++17
- OpenCV
- ONNX Runtime
- gRPC
- Protobuf
- CMake
- vcpkg
- Standard C++ threading primitives

## Project Structure

```text
camera_cv_service/
|-- include/
|   |-- camera_manager.h
|   |-- grpc_server.h
|   |-- inference_engine.h
|   |-- generated_models/
|   `-- models/
|       `-- frame.h
|-- models/
|   `-- detection.proto
|-- src/
|   |-- camera_manager.cpp
|   |-- grpc_server.cpp
|   |-- inference_engine.cpp
|   `-- main.cpp
|-- CMakeLists.txt
`-- README.md
```

## Build Notes

The current `CMakeLists.txt` is still tied to the original Windows development environment.

In particular, it currently assumes machine-specific paths such as:

- `E:/tools/onnxruntime`
- `E:/tools/vcpkg`
- `E:/Progects/test/camera_cv_service`

Before building on another machine, update `CMakeLists.txt` so these paths match your environment.

The project also checks in generated gRPC service files under `include/generated_models/`.
For the current branch, you do not need to regenerate protobuf files unless you modify `models/detection.proto`.

## Example Windows Setup

Typical dependency setup with `vcpkg` may look like this:

```powershell
cd E:\tools\vcpkg
.\vcpkg install opencv4[core,videoio]:x64-windows
.\vcpkg install protobuf:x64-windows
.\vcpkg install grpc:x64-windows
```

ONNX Runtime may be installed manually, depending on your local setup.

Then configure and build:

```powershell
cd E:\Progects\test\camera_cv_service
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=E:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

Expected executable output:

```text
build/bin/Release/camera_cv_service.exe
```

## Running the Service

By default, the service:

- looks for the model at `models/yolov8x.onnx`
- tries to resolve an optional `test_video.mp4`
- auto-detects local cameras from index `0` to `9`
- starts the gRPC server on `0.0.0.0:50051`

Run:

```powershell
camera_cv_service.exe
```

Stop with `Ctrl+C`.

## Client Expectations

A client can subscribe to both streams independently.

For example, conceptually:

```ts
const live = grpcClient.StreamLiveFrames({});
live.on("data", (frame) => {
  // Display immediate JPEG frame
});

const detections = grpcClient.StreamDetectionFrames({});
detections.on("data", (frame) => {
  // Display processed JPEG frame with aligned detections
});
```

## Known Limitations

- `CMakeLists.txt` still contains Windows-specific absolute paths
- the repository has no automated tests yet
- local performance depends heavily on camera resolution, JPEG encoding cost, and YOLO model size
- dual streaming increases network and JPEG encoding load compared with a single-stream design

## Author

- Author: Yurii H.
- Repository: `YuraGB/C-_camera_yolo_service`
