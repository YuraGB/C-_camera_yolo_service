# Runtime Assets

Large runtime files are intentionally not committed to git.

Expected local layout:

```text
models/yolov8x.onnx
media/test_video.mp4
```

`models/yolov8x.onnx` is required. `media/test_video.mp4` is optional when a live camera is available.

## Prepare On Linux

Place the files manually, or provide download URLs through environment variables:

```bash
export CAMERA_MODEL_URL="https://example.com/path/to/yolov8x.onnx"
export CAMERA_TEST_VIDEO_URL="https://example.com/path/to/test_video.mp4"
bash scripts/prepare-assets.sh
```

Without URLs, the script validates that the required files already exist.

## Prepare On Windows

```powershell
$env:CAMERA_MODEL_URL = "https://example.com/path/to/yolov8x.onnx"
$env:CAMERA_TEST_VIDEO_URL = "https://example.com/path/to/test_video.mp4"
.\scripts\prepare-assets.ps1
```

## Docker Mounts

For Docker, mount the asset folders into the Linux container:

```bash
docker run --rm -it \
  --network host \
  --device /dev/video0:/dev/video0 \
  -v /absolute/path/to/models:/models:ro \
  -v /absolute/path/to/media:/media:ro \
  camera-cv-service:linux
```

Inside the container the service reads:

```text
/models/yolo26x.onnx
/media/test_video.mp4
```
