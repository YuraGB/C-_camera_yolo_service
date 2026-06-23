# Camera CV Service

`Camera CV Service` is a C++ service that:

- reads frames from cameras or video files
- runs YOLO inference through ONNX Runtime
- sends low-latency live video to the browser over a native WebRTC video track
- sends detection metadata over a separate WebRTC DataChannel
- sends throttled live-video latency samples over the same DataChannel

The service is split into two independent pipelines:

- `video pipeline`
  - real-time
  - optimized for minimum latency
  - not blocked by slower YOLO work
- `detection pipeline`
  - asynchronous
  - may lag behind live video
  - sends compact detection payloads
- `latency sampling`
  - independent from YOLO
  - samples the live video path at a configurable interval
  - estimates capture-to-browser-render latency on the frontend

## Transport Model

The current transport stack uses:

- `libdatachannel` for WebRTC peer connections
- an external WebSocket signaling server for SDP and ICE exchange
- a native WebRTC `H264` video track for `liveStream`
- a WebRTC `DataChannel` for `detectionStream`
- `OpenH264` loaded at runtime for live video encoding

This keeps the live stream as close to real time as possible while letting detections arrive independently.

## End-to-End Live Path

For live WebRTC playback, the frame path is:

- `file -> OpenCV -> raw frame -> re-encode -> RTP/WebRTC -> browser jitter buffer -> decoder -> screen`

This is important because the service is not acting like a simple local media player. It rebuilds the video into a real-time WebRTC stream, which adds unavoidable processing, packetization, buffering, and decode stages.

## Live Stream

Live video is sent as a native WebRTC video track labeled `liveStream`.

Current behavior:

- frames come from the latest available camera or video frame
- frames are encoded with `OpenH264`
- encoded NAL units are packetized into RTP and sent through the WebRTC track
- if the system cannot keep up, older live frames are dropped instead of building delay

This is intentional: the browser should see the newest frame, not a delayed queue.

## DataChannel Messages

Detection metadata and live-video latency samples are sent through a WebRTC DataChannel labeled `detectionStream`.

Detection payload:

```json
{
  "type": "detection_frame",
  "camera_id": "camera_0",
  "timestamp": 12.533,
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

Live-video latency sample payload:

```json
{
  "type": "video_latency_sample",
  "camera_id": "camera_0",
  "frame_id": 12345,
  "capture_timestamp_ms": 1715000000000,
  "encoded_timestamp_ms": 1715000000024,
  "sample_interval_ms": 1000
}
```

Notes:

- `timestamp` is emitted in relative seconds from service start
- detections may arrive later than the corresponding live video frame
- slow detection must not stall live playback
- `video_latency_sample` is generated from the live video path, not the YOLO path
- latency samples are throttled by `CAMERA_VIDEO_LATENCY_SAMPLE_INTERVAL_MS`
- the frontend uses `requestVideoFrameCallback` to estimate capture-to-render latency

Pipeline metrics payload:

```json
{
  "type": "pipeline_metrics",
  "scope": "video",
  "camera_id": "camera_0",
  "interval_ms": 1000,
  "capture_fps": 29.8,
  "encode_fps": 29.8,
  "avg_capture_delay_ms": 4.2,
  "max_capture_delay_ms": 12,
  "avg_h264_encode_ms": 3.1,
  "max_h264_encode_ms": 7.8,
  "dropped_stale_frames": 0,
  "total_dropped_stale_frames": 0,
  "estimated_live_fps": 29.7
}
```

YOLO inference metrics payload:

```json
{
  "type": "pipeline_metrics",
  "scope": "inference",
  "interval_ms": 1000,
  "submitted_frames": 30,
  "dropped_pending_frames": 20,
  "processed_frames": 8,
  "total_detections": 14,
  "avg_inference_ms": 76.3,
  "max_inference_ms": 91.2
}
```

Metrics are aggregated in memory and published at a low rate. They should not log or serialize per frame.

## Signaling

The service expects an external WebSocket signaling server.

It supports:

- peer registration
- viewer join / offer request messages
- local offer creation
- remote offer handling
- remote answer handling
- ICE candidate exchange

Common signaling message shapes:

```json
{
  "type": "register",
  "peerId": "camera-cv-service"
}
```

```json
{
  "type": "viewer-join",
  "peerId": "frontend-abc",
  "targetPeerId": "camera-cv-service"
}
```

```json
{
  "type": "offer",
  "peerId": "camera-cv-service",
  "targetPeerId": "browser-client",
  "sdp": "..."
}
```

```json
{
  "type": "ice-candidate",
  "peerId": "camera-cv-service",
  "targetPeerId": "browser-client",
  "candidate": "...",
  "mid": "0"
}
```

The service can either:

- create the offer itself after a `viewer-join`, `offer-request`, or `connect` message
- or receive an incoming offer and answer it with the live track plus detection channel

## Runtime Configuration

Main runtime configuration lives in [runtime_config.h](/E:/Progects/test/camera_cv_service/include/core/pipeline/runtime_config.h) and [webrtc_service.h](/E:/Progects/test/camera_cv_service/include/webrtc_service.h).

## Source Layout

The service now has a platform boundary around OS-specific behavior:

```text
include/core/pipeline/      runtime configuration and pipeline-level orchestration
include/platform/           cross-platform service contracts
include/platform/windows/   Windows implementation
include/platform/linux/     Linux implementation
include/capture/            capture-source abstraction
src/platform/windows/       Windows camera/path implementation
src/platform/linux/         Linux camera/path implementation
src/capture/                OpenCV-backed capture source
```

The current concrete platform API covers:

- executable/source-root path discovery
- camera index enumeration
- default signaling URL
- default OpenH264 runtime library name

Windows probes cameras through OpenCV with DirectShow first. Linux probes `/dev/video*` and verifies devices through OpenCV/V4L2.

Local configuration templates:

- `.env.example` is safe to commit
- `.env` is local-only and ignored by git
- large runtime assets are documented in [ASSETS.md](/E:/Progects/test/camera_cv_service/ASSETS.md)

The executable reads environment variables from the process environment. In PowerShell, load `.env` before running:

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -and -not $_.TrimStart().StartsWith("#")) {
    $name, $value = $_ -split "=", 2
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
  }
}
.\build\bin\Release\camera_cv_service.exe
```

When connecting to `server_for_cam_det`, set `CAMERA_AUTH_JWT_SECRET` to the same strong secret as the server's `AUTH_JWT_SECRET`. If the secret is set, the C++ service signs a short-lived HS256 JWT and sends it to the signaling server as the WebSocket `token` query parameter.

Prepare local assets after cloning:

```powershell
.\scripts\prepare-assets.ps1
```

On Linux:

```bash
bash scripts/prepare-assets.sh
```

## Deployment & Runtime Selection

The service supports multiple inference backends:

- **ONNX Runtime** (default) — CPU or GPU via execution providers (CUDA, ROCM, CoreML)
- **TensorRT** (NVIDIA GPU only) — optional, compile-time enabled with `USE_TENSORRT=ON`

### CPU Only (No GPU)

**Docker:**

```bash
docker-compose up --build
```

**Local build:**

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime \
  -DUSE_TENSORRT=OFF

cmake --build build --parallel
cmake --install build --prefix /opt/app

RUNTIME_BACKEND=onnx /opt/app/bin/camera_cv_service
```

**Environment:**

```bash
export RUNTIME_BACKEND=onnx
export CAMERA_MODEL_PATH=/path/to/yolov8x.onnx
export CAMERA_SIGNALING_URL=ws://127.0.0.1:3002/ws
```

### Non-NVIDIA GPU (Radeon/AMD, Intel Arc, etc.)

Use ONNX Runtime GPU providers: ROCM (AMD), OpenCL (Intel), or CPU fallback.

**Docker (CPU fallback):**

```bash
RUNTIME_BACKEND=onnx docker-compose up --build
```

**Docker with ROCM (AMD Radeon):**

```bash
RUNTIME_BACKEND=onnx docker-compose -f docker-compose.yml -f docker-compose.rocm.yml up --build
```

Create `docker-compose.rocm.yml`:

```yaml
version: "3"
services:
  camera-cv:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        ONNXRUNTIME_FLAVOR: rocm
    environment:
      - RUNTIME_BACKEND=onnx
      - CAMERA_MODEL_PATH=/models/yolov8x.onnx
      - CAMERA_SIGNALING_URL=ws://127.0.0.1:3002/ws
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    volumes:
      - ./models:/models
      - ./media:/media
```

Run:

```bash
docker-compose -f docker-compose.rocm.yml up --build
```

**Local build for AMD ROCM:**

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime-rocm \
  -DUSE_TENSORRT=OFF

cmake --build build --parallel

RUNTIME_BACKEND=onnx ./build/bin/camera_cv_service
```

### NVIDIA GPU (TensorRT Recommended)

**Docker with TensorRT (recommended for NVIDIA):**

```bash
docker compose -f docker-compose.tensorrt.yml build
docker compose -f docker-compose.tensorrt.yml up
```

Or build and start in one command:

```bash
docker-compose -f docker-compose.tensorrt.yml up --build
```

This uses:

- NVIDIA CUDA base image
- TensorRT 8.6+ runtime
- ONNX Runtime with CUDA support
- Automatic GPU device mapping

**Environment:**

```bash
export RUNTIME_BACKEND=tensorrt
export CAMERA_MODEL_PATH=/models/yolov8x.engine
export CAMERA_ONNX_FALLBACK_MODEL_PATH=/models/yolov8x.onnx
export CAMERA_SIGNALING_URL=ws://127.0.0.1:3002/ws
```

**Local build for NVIDIA with TensorRT:**

Requirements:

- CUDA Toolkit 12.2+
- TensorRT 8.6+
- nvidia-docker or `--gpus all` support

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime \
  -DUSE_TENSORRT=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DTENSORRT_INCLUDE_DIR=/usr/include

cmake --build build --parallel
cmake --install build --prefix /opt/app

RUNTIME_BACKEND=tensorrt /opt/app/bin/camera_cv_service
```

**Model conversion (ONNX → TensorRT engine):**

```bash
trtexec \
  --onnx=yolov8x.onnx \
  --saveEngine=yolov8x.engine \
  --workspace=8192 \
  --fp16
```

Then set:

```bash
export CAMERA_MODEL_PATH=/path/to/yolov8x.engine
export CAMERA_ONNX_FALLBACK_MODEL_PATH=/path/to/yolov8x.onnx
export RUNTIME_BACKEND=tensorrt
```

### Runtime Backend Selection

The `RUNTIME_BACKEND` environment variable selects the inference engine:

- `onnx` (default) — ONNX Runtime
- `tensorrt` — TensorRT (only if compiled with `USE_TENSORRT=ON`)

If TensorRT is requested but fails to initialize, the service automatically falls back to ONNX Runtime.

```bash
RUNTIME_BACKEND=tensorrt ./camera_cv_service
# Falls back to ONNX if TensorRT unavailable
```

Useful environment variables:

- `CAMERA_MODEL_PATH`
- `CAMERA_ONNX_FALLBACK_MODEL_PATH`
- `CAMERA_TEST_VIDEO_PATH`
- `CAMERA_SIGNALING_URL`
- `CAMERA_PEER_ID`
- `CAMERA_REMOTE_PEER_ID`
- `CAMERA_AUTH_JWT_SECRET`
- `CAMERA_AUTH_JWT_ISSUER`
- `CAMERA_AUTH_JWT_AUDIENCE`
- `CAMERA_AUTH_JWT_ROLE`
- `CAMERA_AUTH_JWT_EMAIL`
- `CAMERA_AUTH_JWT_TTL_SECONDS`
- `CAMERA_MAX_CAMERA_SCAN`
- `CAMERA_INFERENCE_WIDTH`
- `CAMERA_INFERENCE_HEIGHT`
- `CAMERA_CONF_THRESHOLD`
- `CAMERA_IOU_THRESHOLD`
- `CAMERA_H264_BITRATE_BPS`
- `CAMERA_OPENH264_LIBRARY`
- `CAMERA_MAX_LIVE_LATENCY_MS`
- `CAMERA_MAX_LIVE_WIDTH`
- `CAMERA_MAX_LIVE_HEIGHT`
- `CAMERA_VIDEO_LATENCY_SAMPLE_INTERVAL_MS`
- `CAMERA_PIPELINE_METRICS_INTERVAL_MS`
- `CAMERA_MAX_DETECTION_BUFFERED_BYTES`
- `CAMERA_VERBOSE_LOGS`

Defaults:

- signaling URL: `ws://127.0.0.1:3002/ws`
- local peer id: `camera-cv-service`
- JWT issuer/audience: `camera-cv-service` / `signaling`
- JWT TTL: `300s`
- ICE server: `stun:stun.l.google.com:19302`
- inference size: `640x640`
- H264 bitrate: `2500000`
- OpenH264 library:
  - Windows: `openh264-2.6.0-win64.dll`
  - Linux: `libopenh264.so.2`
- max live latency: `150ms`
- live-video latency sample interval: `1000ms`
- pipeline metrics interval: `1000ms`

## Build Dependencies

- C++17
- OpenCV
- ONNX Runtime 1.18
- libdatachannel
- OpenSSL
- OpenH264 runtime library
- vcpkg

## Build On Windows

```powershell
cd E:\Progects\test\camera_cv_service
cmake -S . -B build
cmake --build build --config Release
```

Binary:

```text
build/bin/Release/camera_cv_service.exe
```

## Build On Linux

Install OpenCV, libdatachannel, Eigen3, OpenSSL, OpenH264, and ONNX Runtime for your distro or CI image. Configure `ONNXRUNTIME_ROOT` when ONNX Runtime is not installed under `/opt/onnxruntime`.

```bash
cd /path/to/camera_cv_service
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/opt/onnxruntime
cmake --build build --config Release
```

Linux runtime notes:

- OpenH264 is loaded dynamically with `dlopen`
- default OpenH264 library is `libopenh264.so.7` in the Docker image
- override with `CAMERA_OPENH264_LIBRARY=/absolute/path/to/libopenh264.so.7` when needed
- camera discovery scans `/dev/video*` and validates indices with V4L2/OpenCV

## Docker On Linux

The root [Dockerfile](/E:/Progects/test/camera_cv_service/Dockerfile) builds a Linux runtime image with:

- Ubuntu 24.04 base
- ONNX Runtime under `/opt/onnxruntime`
- OpenH264 built as a Linux shared library
- libdatachannel built from source
- default model mount path `/models/yolo26x.onnx`

Build and run the default CPU image with Compose:

```bash
docker compose build
docker compose up
```

GPU support is opt-in. Use the GPU override only on hosts with a working NVIDIA Docker runtime:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

Run with a USB/V4L2 camera and host networking:

```bash
docker run --rm -it \
  --network host \
  --device /dev/video0:/dev/video0 \
  --group-add video \
  -v /absolute/path/to/models:/models:ro \
  -e CAMERA_SIGNALING_URL=ws://127.0.0.1:3002/ws \
  -e CAMERA_AUTH_JWT_SECRET="$AUTH_JWT_SECRET" \
  camera-cv-service:latest
```

Run with NVIDIA GPU access:

```bash
docker run --rm -it \
  --network host \
  --gpus all \
  --device /dev/video0:/dev/video0 \
  --group-add video \
  -v /absolute/path/to/models:/models:ro \
  -e CAMERA_SIGNALING_URL=ws://127.0.0.1:3002/ws \
  -e CAMERA_AUTH_JWT_SECRET="$AUTH_JWT_SECRET" \
  camera-cv-service:latest
```

Container notes:

- use `--network host` when the signaling server is also on the Linux host
- mount large models and test videos instead of baking them into the image
- the default Compose file does not request GPU, so hosts without GPU run on CPU
- the GPU override requires Docker to discover a supported GPU before the container starts
- set `CAMERA_TEST_VIDEO_PATH=/media/<file>` and mount `-v /path/to/media:/media:ro` for file playback

## Runtime Notes

- the executable expects `models/yolov8x.onnx`
- an optional `media/test_video.mp4` is used automatically when present
- legacy `test_video.mp4` beside the executable is still checked as a fallback
- `openh264-2.6.0-win64.dll` is copied into the runtime output during the build
- on Linux, OpenH264 is expected to be available through the system loader or `CAMERA_OPENH264_LIBRARY`
- ONNX Runtime 1.18 GPU loading still depends on a matching CUDA/cuDNN runtime on the machine
- GPU execution is attempted first when CUDA is available; CPU remains the fallback

## Current Design Constraints

- live video uses native WebRTC media transport
- detection metadata uses a DataChannel, not a second media track
- detection timing is intentionally decoupled from live playback
- live-video latency sampling is intentionally decoupled from detection timing
- exact frame-to-sample correlation is best-effort because browser `<video>` does not expose RTP frame ids directly
- if the browser or network is slower than capture, the service prefers dropping outdated live frames over increasing latency
