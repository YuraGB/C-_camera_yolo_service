#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resolve_path() {
  local raw_path="$1"
  if [[ "${raw_path}" = /* ]]; then
    printf '%s\n' "${raw_path}"
  else
    printf '%s/%s\n' "${ROOT_DIR}" "${raw_path}"
  fi
}

MODEL_PATH="$(resolve_path "${CAMERA_MODEL_PATH:-models/yolov8x.onnx}")"
TEST_VIDEO_PATH="$(resolve_path "${CAMERA_TEST_VIDEO_PATH:-media/test_video.mp4}")"
MODEL_URL="${CAMERA_MODEL_URL:-}"
TEST_VIDEO_URL="${CAMERA_TEST_VIDEO_URL:-}"

download_if_needed() {
  local label="$1"
  local url="$2"
  local destination="$3"

  if [[ -f "${destination}" ]]; then
    return 0
  fi

  if [[ -z "${url}" ]]; then
    return 1
  fi

  mkdir -p "$(dirname "${destination}")"
  echo "[assets] Downloading ${label} -> ${destination}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "${destination}.tmp" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${destination}.tmp" "${url}"
  else
    echo "[assets] ERROR: install curl or wget to download ${label}" >&2
    return 1
  fi
  mv "${destination}.tmp" "${destination}"
}

require_file() {
  local label="$1"
  local path="$2"

  if [[ ! -s "${path}" ]]; then
    echo "[assets] ERROR: ${label} is missing or empty: ${path}" >&2
    return 1
  fi
}

mkdir -p "${ROOT_DIR}/models" "${ROOT_DIR}/media"

download_if_needed "YOLO model" "${MODEL_URL}" "${MODEL_PATH}" || true
if [[ -n "${TEST_VIDEO_URL}" ]]; then
  download_if_needed "test video" "${TEST_VIDEO_URL}" "${TEST_VIDEO_PATH}" || true
fi

require_file "YOLO model" "${MODEL_PATH}"

if [[ ! -s "${TEST_VIDEO_PATH}" ]]; then
  echo "[assets] WARN: optional test video is missing: ${TEST_VIDEO_PATH}"
fi

echo "[assets] OK"
echo "[assets] model: ${MODEL_PATH}"
echo "[assets] video: ${TEST_VIDEO_PATH}"
echo
echo "Docker mount example:"
echo "  -v \"$(dirname "${MODEL_PATH}"):/models:ro\""
echo "  -v \"$(dirname "${TEST_VIDEO_PATH}"):/media:ro\""
