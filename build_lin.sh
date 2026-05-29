#!/usr/bin/env bash
set -euo pipefail

default_image="roboflow-annotations"
default_container="roboflow-annotations-run"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -r -p "Docker image name [${default_image}]: " image_name
read -r -p "Docker container name [${default_container}]: " container_name
read -r -p "Is the Docker image already built? [y/N]: " use_existing

image_name="${image_name:-$default_image}"
container_name="${container_name:-$default_container}"
use_existing="${use_existing,,}"

docker rm -f "$container_name" >/dev/null 2>&1 || true

if [[ "$use_existing" != "y" ]]; then
    echo "Building Docker image '$image_name'..."
    docker build --network host -t "$image_name" "$script_dir"
fi

gpu_args=()
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_args=(--gpus all)
fi

echo "Running Docker container '$container_name'..."
docker run -it \
    --name "$container_name" \
    --network host \
    -e HOME=/tmp \
    -e USER=root \
    -e LOGNAME=root \
    -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    -e XDG_CACHE_HOME=/tmp/.cache \
    -e PYTHONPYCACHEPREFIX=/tmp/pycache \
    "${gpu_args[@]}" \
    -v "$script_dir:/a2:rw" \
    "$image_name"
