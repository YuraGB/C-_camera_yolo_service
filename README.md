# Camera CV Service

`Camera CV Service` is a C++ service that:

- reads frames from cameras or video files
- runs YOLO inference through ONNX Runtime
- sends low-latency live video to the browser over a native WebRTC video track
- sends detection metadata over a separate WebRTC DataChannel

The service is split into two independent pipelines:

- `video pipeline`
  - real-time
  - optimized for minimum latency
  - not blocked by slower YOLO work
- `detection pipeline`
  - asynchronous
  - may lag behind live video
  - sends compact detection payloads

## Transport Model

The current transport stack uses:

- `libdatachannel` for WebRTC peer connections
- an external WebSocket signaling server for SDP and ICE exchange
- a native WebRTC `H264` video track for `liveStream`
- a WebRTC `DataChannel` for `detectionStream`
- `OpenH264` loaded at runtime for live video encoding

This keeps the live stream as close to real time as possible while letting detections arrive independently.

## Live Stream

Live video is sent as a native WebRTC video track labeled `liveStream`.

Current behavior:

- frames come from the latest available camera or video frame
- frames are encoded with `OpenH264`
- encoded NAL units are packetized into RTP and sent through the WebRTC track
- if the system cannot keep up, older live frames are dropped instead of building delay

This is intentional: the browser should see the newest frame, not a delayed queue.

## Detection Stream

Detection metadata is sent through a separate WebRTC DataChannel labeled `detectionStream`.

Payload format:

```json
{
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

Notes:

- `timestamp` is emitted in relative seconds from service start
- detections may arrive later than the corresponding live video frame
- slow detection must not stall live playback

## Signaling

The service expects an external WebSocket signaling server.

It supports:

- peer registration
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

- create the offer itself after a peer joins
- or receive an incoming offer and answer it with the live track plus detection channel

## Runtime Configuration

Main runtime configuration lives in [webrtc_service.h](/E:/Progects/test/camera_cv_service/include/webrtc_service.h) and [main.cpp](/E:/Progects/test/camera_cv_service/src/main.cpp).

Useful environment variables:

- `CAMERA_SIGNALING_URL`
- `CAMERA_PEER_ID`
- `CAMERA_REMOTE_PEER_ID`

Defaults:

- signaling URL: `ws://127.0.0.1:3001/ws`
- local peer id: `camera-cv-service`
- ICE server: `stun:stun.l.google.com:19302`

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

## Current Design Constraints

- live video uses native WebRTC media transport
- detection metadata uses a DataChannel, not a second media track
- detection timing is intentionally decoupled from live playback
- if the browser or network is slower than capture, the service prefers dropping outdated live frames over increasing latency
