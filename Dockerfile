# syntax=docker/dockerfile:1.7

ARG UBUNTU_VERSION=24.04
ARG ONNXRUNTIME_VERSION=1.18.1
ARG ONNXRUNTIME_FLAVOR=gpu
ARG LIBDATACHANNEL_VERSION=v0.22.5

FROM ubuntu:${UBUNTU_VERSION} AS build-deps

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_FLAVOR
ARG LIBDATACHANNEL_VERSION

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        cmake \
        curl \
        git \
        make \
        nasm \
        ninja-build \
        pkg-config \
        tar \
        xz-utils \
        libopencv-dev \
        libssl-dev \
        libx11-dev \
        libxext-dev \
        libxrender-dev \
        libv4l-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN case "${ONNXRUNTIME_FLAVOR}" in \
        gpu) package="onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}" ;; \
        cpu) package="onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" ;; \
        *) echo "Unsupported ONNXRUNTIME_FLAVOR=${ONNXRUNTIME_FLAVOR}; use gpu or cpu" >&2; exit 1 ;; \
    esac \
    && curl -fsSL \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${package}.tgz" \
        -o /tmp/onnxruntime.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm -f /tmp/onnxruntime.tgz

RUN git clone --depth 1 --branch "${LIBDATACHANNEL_VERSION}" \
        https://github.com/paullouisageneau/libdatachannel.git /tmp/libdatachannel \
    && cmake -S /tmp/libdatachannel -B /tmp/libdatachannel/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DNO_EXAMPLES=ON \
        -DNO_TESTS=ON \
        -DUSE_GNUTLS=OFF \
        -DUSE_NICE=OFF \
    && cmake --build /tmp/libdatachannel/build --parallel \
    && cmake --install /tmp/libdatachannel/build \
    && ldconfig \
    && rm -rf /tmp/libdatachannel

FROM build-deps AS builder

WORKDIR /src
COPY . .

RUN make -C third_party/openh264-2.6.0 -j"$(nproc)" BUILDTYPE=Release PREFIX=/opt/openh264 install \
    && cmake -S . -B /tmp/camera_cv_service_build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DONNXRUNTIME_ROOT=/opt/onnxruntime \
    && cmake --build /tmp/camera_cv_service_build --parallel \
    && install -Dm755 \
        /tmp/camera_cv_service_build/bin/camera_cv_service \
        /opt/camera_cv_service/bin/camera_cv_service

FROM ubuntu:${UBUNTU_VERSION} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libopencv-dev \
        openssl \
        libstdc++6 \
        libgcc-s1 \
        libgomp1 \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system camera \
    && (getent group video >/dev/null || groupadd --system video) \
    && useradd --system --create-home --gid camera --groups video --home-dir /app camera

COPY --from=build-deps /opt/onnxruntime /opt/onnxruntime
COPY --from=build-deps /usr/local /usr/local
COPY --from=builder /opt/openh264 /opt/openh264
COPY --from=builder /opt/camera_cv_service /opt/camera_cv_service

RUN ldconfig \
    && mkdir -p /models /media /app \
    && chown -R camera:camera /app /models /media

ENV LD_LIBRARY_PATH="/opt/onnxruntime/lib:/opt/openh264/lib:/usr/local/lib" \
    CAMERA_MODEL_PATH="/models/yolov8x.onnx" \
    CAMERA_TEST_VIDEO_PATH="/media/test_video.mp4" \
    CAMERA_SIGNALING_URL="ws://127.0.0.1:3001/ws" \
    CAMERA_PEER_ID="camera-cv-service" \
    CAMERA_OPENH264_LIBRARY="/opt/openh264/lib/libopenh264.so.2" \
    CAMERA_MAX_CAMERA_SCAN="10" \
    CAMERA_MAX_LIVE_LATENCY_MS="150" \
    CAMERA_VIDEO_LATENCY_SAMPLE_INTERVAL_MS="1000" \
    CAMERA_PIPELINE_METRICS_INTERVAL_MS="1000" \
    CAMERA_MAX_DETECTION_BUFFERED_BYTES="131072" \
    CAMERA_VERBOSE_LOGS="false"

WORKDIR /app
USER camera

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/camera_cv_service/bin/camera_cv_service"]
