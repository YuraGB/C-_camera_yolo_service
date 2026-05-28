# syntax=docker/dockerfile:1.7

ARG UBUNTU_VERSION=24.04
ARG ONNXRUNTIME_VERSION=1.18.1
ARG ONNXRUNTIME_FLAVOR=gpu
ARG LIBDATACHANNEL_VERSION=v0.22.5

############################
# BASE BUILD IMAGE
############################
FROM ubuntu:${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    cmake \
    curl \
    git \
    make \
    meson \
    ninja-build \
    pkg-config \
    nasm \
    tar \
    xz-utils \
    libopencv-dev \
    libssl-dev \
    libv4l-dev \
    zlib1g-dev \
    libeigen3-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

############################
# ONNXRUNTIME
############################
FROM base AS onnxruntime

ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_FLAVOR

RUN case "${ONNXRUNTIME_FLAVOR}" in \
    gpu) pkg="onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}" ;; \
    cpu) pkg="onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" ;; \
    *) echo "bad flavor" && exit 1 ;; \
    esac \
    && curl -fsSL \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${pkg}.tgz" \
        -o /tmp/ort.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm -f /tmp/ort.tgz

############################
# libdatachannel
############################
FROM base AS libdatachannel

ARG LIBDATACHANNEL_VERSION

WORKDIR /tmp

RUN git clone --recursive --depth 1 --branch ${LIBDATACHANNEL_VERSION} \
    https://github.com/paullouisageneau/libdatachannel.git

RUN cmake -S libdatachannel -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DNO_EXAMPLES=ON \
    -DNO_TESTS=ON \
    -DUSE_GNUTLS=OFF \
    -DUSE_NICE=OFF \
    && cmake --build build --parallel \
    && cmake --install build --prefix /opt/libdatachannel \
    && rm -rf /tmp/libdatachannel

############################
# APP BUILD
############################
FROM base AS builder

WORKDIR /src

COPY . .

RUN git submodule update --init --recursive || true

RUN if [ ! -f third_party/ByteTrack-cpp/src/BYTETracker.cpp ]; then \
        git clone --depth 1 https://github.com/Vertical-Beach/ByteTrack-cpp.git third_party/ByteTrack-cpp; \
    fi

COPY --from=onnxruntime /opt/onnxruntime /opt/onnxruntime
COPY --from=libdatachannel /opt/libdatachannel /usr/local

############################
# OpenH264
############################
RUN if [ -f third_party/openh264-2.6.0/codec/common/x86/cpuid.asm ]; then \
        openh264_source=third_party/openh264-2.6.0; \
    else \
        git clone --depth 1 --branch v2.6.0 https://github.com/cisco/openh264.git /tmp/openh264; \
        openh264_source=/tmp/openh264; \
    fi \
    && meson setup /tmp/openh264-build "${openh264_source}" \
        --prefix=/opt/openh264 \
        --libdir=lib \
        --buildtype=release \
        -Dtests=disabled \
    && meson compile -C /tmp/openh264-build \
    && meson install -C /tmp/openh264-build \
    && rm -rf /tmp/openh264 /tmp/openh264-build

############################
# MAIN BUILD
############################
RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT=/opt/onnxruntime \
    -DCMAKE_PREFIX_PATH="/usr/local;/opt/openh264" \
    && cmake --build build --parallel \
    && cmake --install build --prefix /opt/app

############################
# RUNTIME
############################
FROM ubuntu:${UBUNTU_VERSION} AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    libstdc++6 \
    libgcc-s1 \
    libgomp1 \
    libssl3 \
    openssl \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=onnxruntime /opt/onnxruntime /opt/onnxruntime
COPY --from=builder /opt/openh264 /opt/openh264
COPY --from=builder /opt/app /opt/app
COPY --from=libdatachannel /opt/libdatachannel /usr/local

ENV LD_LIBRARY_PATH="/opt/onnxruntime/lib:/opt/openh264/lib:/usr/local/lib" \
    CAMERA_OPENH264_LIBRARY="/opt/openh264/lib/libopenh264.so.7"

WORKDIR /app

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/app/bin/camera_cv_service"]
