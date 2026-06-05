#!/usr/bin/env bash
set -euo pipefail

IMAGE="ethos"
IMAGE_FULL="ethos-full"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
UID_VAL="$(id -u)"
GID_VAL="$(id -g)"

usage() {
    cat <<'EOF'
Usage: ethos-docker.sh <command> [options]

Commands:
  build [--vllm]         Build the Docker image (optional: include vLLM)
  shell                  Interactive bash shell in the container
  <ethos cmd>         Pass through to ethos CLI
                         (ablate, talk, test, list, tui, setup, etc.)

Examples:
  ethos-docker.sh build
  ethos-docker.sh build --vllm
  ethos-docker.sh ablate --model Qwen/Qwen2.5-7B-Instruct --out qwen-ethos
  ethos-docker.sh talk --model qwen-ethos --quant nf4
  ethos-docker.sh test --model qwen-ethos --base Qwen/Qwen2.5-7B-Instruct --suite all
  ethos-docker.sh shell

Environment:
  HF_CACHE     HuggingFace cache dir (default: ~/.cache/huggingface)
  HF_TOKEN     HuggingFace token (passed through if set)
  ETHOS_GPU GPU device(s) (default: all)
EOF
    exit 0
}

docker_image() {
    if docker image inspect "$IMAGE_FULL" >/dev/null 2>&1; then
        echo "$IMAGE_FULL"
    else
        echo "$IMAGE"
    fi
}

run_args() {
    local -a args=(
        --rm
        --gpus "${ETHOS_GPU:-all}"
        --shm-size=8g
        --user "$UID_VAL:$GID_VAL"
        -v "$PWD:/host"
        -v "$HF_CACHE:/home/ethos/.cache/huggingface"
        -e "HOME=/home/ethos"
        -e "PYTHONUNBUFFERED=1"
        -w /host
    )

    if [ -n "${HF_TOKEN:-}" ]; then
        args+=(-e "HF_TOKEN=$HF_TOKEN")
    fi

    if [ -t 0 ]; then
        args+=(-it)
    fi

    echo "${args[@]}"
}

cmd_build() {
    local target="ethos"
    local tag="$IMAGE"
    local -a extra_args=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --vllm)
                target="ethos-vllm"
                tag="$IMAGE_FULL"
                ;;
            *)
                extra_args+=("$1")
                ;;
        esac
        shift
    done

    echo "Building $tag (target: $target, UID: $UID_VAL, GID: $GID_VAL) ..."
    docker build \
        --target "$target" \
        --build-arg "USER_ID=$UID_VAL" \
        --build-arg "GROUP_ID=$GID_VAL" \
        -t "$tag" \
        "${extra_args[@]}" \
        "$(dirname "$0")"
}

cmd_shell() {
    local img
    img="$(docker_image)"
    # shellcheck disable=SC2046
    docker run $(run_args) --entrypoint bash "$img"
}

cmd_run() {
    local img
    img="$(docker_image)"

    if echo "$*" | grep -q -- '--backend vllm'; then
        if [ "$img" != "$IMAGE_FULL" ]; then
            echo "WARNING: --backend vllm requested but image '$IMAGE_FULL' not found."
            echo "  Run: ethos-docker.sh build --vllm"
        fi
    fi

    # shellcheck disable=SC2046
    docker run $(run_args) "$img" "$@"
}

if [ $# -eq 0 ]; then
    cmd_run
else
    cmd="$1"
    shift
    case "$cmd" in
        build)
            cmd_build "$@"
            ;;
        shell)
            cmd_shell
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            cmd_run "$cmd" "$@"
            ;;
    esac
fi
