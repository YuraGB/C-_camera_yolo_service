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

Main runtime configuration lives in [webrtc_service.h](/E:/Progects/test/camera_cv_service/include/webrtc_service.h) and [main.cpp](/E:/Progects/test/camera_cv_service/src/main.cpp).

Local configuration templates:

- `.env.example` is safe to commit
- `.env` is local-only and ignored by git

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

Useful environment variables:

- `CAMERA_MODEL_PATH`
- `CAMERA_TEST_VIDEO_PATH`
- `CAMERA_SIGNALING_URL`
- `CAMERA_PEER_ID`
- `CAMERA_REMOTE_PEER_ID`
- `CAMERA_MAX_CAMERA_SCAN`
- `CAMERA_INFERENCE_WIDTH`
- `CAMERA_INFERENCE_HEIGHT`
- `CAMERA_CONF_THRESHOLD`
- `CAMERA_IOU_THRESHOLD`
- `CAMERA_H264_BITRATE_BPS`
- `CAMERA_MAX_LIVE_LATENCY_MS`
- `CAMERA_MAX_LIVE_WIDTH`
- `CAMERA_MAX_LIVE_HEIGHT`
- `CAMERA_VIDEO_LATENCY_SAMPLE_INTERVAL_MS`
- `CAMERA_PIPELINE_METRICS_INTERVAL_MS`
- `CAMERA_MAX_DETECTION_BUFFERED_BYTES`
- `CAMERA_VERBOSE_LOGS`

Defaults:

- signaling URL: `ws://127.0.0.1:3001/ws`
- local peer id: `camera-cv-service`
- ICE server: `stun:stun.l.google.com:19302`
- inference size: `640x640`
- H264 bitrate: `2500000`
- max live latency: `150ms`
- live-video latency sample interval: `1000ms`
- pipeline metrics interval: `1000ms`

## Build Dependencies

- C++17
- OpenCV
- ONNX Runtime 1.18
- libdatachannel
- OpenH264 runtime DLL
- vcpkg

## Build

```powershell
cd E:\Progects\test\camera_cv_service
cmake -S . -B build
cmake --build build --config Release
```

Binary:

```text
build/bin/Release/camera_cv_service.exe
```

## Runtime Notes

- the executable expects `models/yolov8x.onnx`
- an optional `test_video.mp4` beside the executable is used automatically when present
- `openh264-2.6.0-win64.dll` is copied into the runtime output during the build
- ONNX Runtime 1.18 GPU loading still depends on a matching CUDA/cuDNN runtime on the machine
- GPU execution is attempted first when CUDA is available; CPU remains the fallback

## Current Design Constraints

- live video uses native WebRTC media transport
- detection metadata uses a DataChannel, not a second media track
- detection timing is intentionally decoupled from live playback
- live-video latency sampling is intentionally decoupled from detection timing
- exact frame-to-sample correlation is best-effort because browser `<video>` does not expose RTP frame ids directly
- if the browser or network is slower than capture, the service prefers dropping outdated live frames over increasing latency
