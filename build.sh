#!/usr/bin/env bash
# ==============================================================================
# WEDA-LLM-BASE Pipeline — Build & Launch Script
#
# Usage:
#   ./build.sh              # Build/Pull, start, and enter the container
#   ./build.sh --run        # Start only (skip build/pull, use existing image)
#   ./build.sh --shell      # Attach to already-running container
#   ./build.sh --stop       # Stop and remove the container
#   ./build.sh --test       # Run the container diagnostics (scripts/wise-bench)
#   ./build.sh --help
# ==============================================================================
set -euo pipefail
IFS=$'\n\t'

# ── Path Anchoring ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
readonly SCRIPT_DIR
cd "${SCRIPT_DIR}"

# ── Logging ───────────────────────────────────────────────────────────────────
log_info()    { echo -e "\e[1;36m[INFO]\e[0m  $1"; }
log_success() { echo -e "\e[1;32m[  OK]\e[0m  $1"; }
log_warn()    { echo -e "\e[1;33m[WARN]\e[0m  $1"; }
log_error()   { echo -e "\e[1;31m[FAIL]\e[0m  $1" >&2; exit 1; }

# ── Shared pre-flight (disk, ports, registry login) ──────────────────────────
# weda_preflight [port ...]: refuse to start into a state the user cannot recover
# from silently -- a full docker root, a port another process already serves,
# or a private-registry pull without credentials. Warnings never abort; errors do.
weda_preflight() {
    local root free_gb port holder
    root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
    free_gb="$(df -BG --output=avail "$root" 2>/dev/null | tail -1 | tr -dc '0-9')"
    if [[ -n "${free_gb}" ]]; then
        if (( free_gb < 5 )); then log_error "Only ${free_gb} GB free under ${root}; images and model caches need more. Free space (docker system df / docker image prune) and retry."; fi
        if (( free_gb < 20 )); then log_warn "${free_gb} GB free under ${root} -- large images or model downloads may fail."; fi
    fi
    for port in "$@"; do
        holder="$(ss -Hltnp 2>/dev/null | awk -v p=":${port}$" '$4 ~ p {print $NF; exit}')"
        if ss -Hltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}$"; then
            log_error "Port ${port} is already in use on this host${holder:+ by ${holder}}. Stop that process/container or change the port in .env, then retry."
        fi
    done
    if [[ -n "${REGISTRY_HOST:-}" ]] && ! grep -q "\"${REGISTRY_HOST}\"" "${HOME}/.docker/config.json" 2>/dev/null; then
        log_warn "Not logged in to ${REGISTRY_HOST}: a pull of a private image will be refused (docker login ${REGISTRY_HOST}). A local image is used when present."
    fi
}

# is_truthy VALUE: same 1/true/yes set (case-insensitive) the python side
# (serve.py, download_models.py) accepts for HF_HUB_OFFLINE.
is_truthy() {
    case "${1,,}" in
        1|true|yes) return 0 ;;
        *) return 1 ;;
    esac
}

# best-effort host-side flush before a load; the in-container flush in
# scripts/serve.py needs elevated container permissions this compose file
# does not grant, so it degrades to a warning there.
# -n so this never blocks on a password prompt: without passwordless sudo
# it fails at once and the caller continues.
flush_page_cache() {
    sudo -n sync && sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null \
        || log_warn "could not flush host page cache (needs passwordless sudo) -- continuing"
}

# ── Parse Flags ───────────────────────────────────────────────────────────────
MODE="build"
for arg in "$@"; do
    case "$arg" in
        --run)    MODE="run" ;;
        --serve)  MODE="serve" ;;
        --shell)  MODE="shell" ;;
        --stop)   MODE="stop" ;;
        --test)   MODE="test" ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "  (no flag)   Build/Pull the container, start it, wait for hydration, open shell"
            echo "  --run       Start the container (skip pull, use existing image)"
            echo "  --serve     Start the container (skip pull), flush memory caches, and exit without opening a shell"
            echo "  --shell     Attach a shell to the already-running container"
            echo "  --stop      Stop and remove the container"
            echo "  --test      Run the container diagnostics (scripts/wise-bench)"
            echo "  --help      Show this help message"
            exit 0 ;;
    esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "\033[1;36m"
cat << "EOF"
   ╦ ╦╔═╗╔╦╗╔═╗   ╦  ╦  ╔╦╗  ╔╗ ╔═╗╔═╗╔═╗
   ║║║║╣  ║║╠═╣───║  ║  ║║║──╠╩╗╠═╣╚═╗║╣
   ╚╩╝╚═╝═╩╝╩ ╩   ╩═╝╩═╝╩ ╩  ╚═╝╩ ╩╚═╝╚═╝
          Jetson Thor Optimized  |  Powered by Advantech Co. Ltd
EOF
echo -e "\033[0m"

# ── Load Config from .env ─────────────────────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/.env"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

[[ -f "${ENV_FILE}" ]] || log_error ".env not found at ${ENV_FILE}"

set -a
# shellcheck source=/dev/null
source "${ENV_FILE}"
set +a
# HF_TOKEN lives in .env.secrets (untracked), not the committed .env
[[ -f "${SCRIPT_DIR}/.env.secrets" ]] && { set -a; source "${SCRIPT_DIR}/.env.secrets"; set +a; }

if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
else
    COMPOSE=()
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    log_warn "HF_TOKEN is not set (put it in .env.secrets next to .env) — gated models will fail to pull."
fi

# ── Static Variable Validation ────────────────────────────────────────────────
REQUIRED_STATIC=(
    WEDA_LLM_IMAGE
    WEDA_LLM_CONTAINER_NAME
    WEDA_LLM_PORT
)

MISSING=()
for var in "${REQUIRED_STATIC[@]}"; do
    [[ -z "${!var:-}" ]] && MISSING+=("${var}")
done
[[ ${#MISSING[@]} -gt 0 ]] && log_error "Missing required vars in .env: ${MISSING[*]}"

# 0.0.0.0 is a bind wildcard, not a reachable probe target
PROBE_HOST="${WEDA_LLM_HOST:-127.0.0.1}"
[[ "${PROBE_HOST}" == "0.0.0.0" || -z "${PROBE_HOST}" ]] && PROBE_HOST="127.0.0.1"
export PROBE_HOST

# ── Setup Volumes ─────────────────────────────────────────────────────────────
HF_CACHE="${HF_CACHE:-${SCRIPT_DIR}/model_cache/huggingface}"
# a shell does not expand ~ inside a variable; compose would create a literal '~' dir
HF_CACHE="${HF_CACHE/#\~/$HOME}"
# a relative HF_CACHE reaches the container un-absolutised otherwise
[[ "${HF_CACHE}" = /* ]] || HF_CACHE="${SCRIPT_DIR}/${HF_CACHE#./}"
export HF_CACHE   # must match the docker-compose volume default so the host pre-check below looks at the same cache the container mounts
MOE_CONFIGS_DIR="${MOE_CONFIGS_DIR:-${SCRIPT_DIR}/artifacts/moe-configs}"
TEST_LOGS_DIR="${TEST_LOGS_DIR:-${SCRIPT_DIR}/test_logs}"

mkdir -p "${HF_CACHE}" "${MOE_CONFIGS_DIR}" "${TEST_LOGS_DIR}"

# ── Handle --stop ─────────────────────────────────────────────────────────────
if [[ "${MODE}" == "stop" ]]; then
    echo -e "\033[1;34m[1/1] Stopping container...\033[0m"
    if docker ps -a -q -f name="^${WEDA_LLM_CONTAINER_NAME}$" | grep -q .; then
        "${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" down --remove-orphans
        log_success "Container stopped and removed."
    else
        log_warn "Container '${WEDA_LLM_CONTAINER_NAME}' is not running."
    fi
    exit 0
fi

# ── Handle --shell ────────────────────────────────────────────────────────────
if [[ "${MODE}" == "shell" ]]; then
    echo -e "\033[1;34m[1/1] Attaching to running container...\033[0m"
    if docker ps -q -f name="^${WEDA_LLM_CONTAINER_NAME}$" | grep -q .; then
        log_success "Entering container '${WEDA_LLM_CONTAINER_NAME}'..."
        docker exec -it "${WEDA_LLM_CONTAINER_NAME}" /bin/bash || true
    else
        log_error "Container '${WEDA_LLM_CONTAINER_NAME}' is not running. Start it first with: ./build.sh"
    fi
    exit 0
fi

# ── Handle --test ─────────────────────────────────────────────────────────────
if [[ "${MODE}" == "test" ]]; then
    echo -e "\033[1;34m[1/1] Running verification tests...\033[0m"
    "${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" run --rm --name "${WEDA_LLM_CONTAINER_NAME}-test" weda-llm \
      /wise-edge/weda-llm-base/scripts/wise-bench \
        --report-dir /wise-edge/weda-llm-base/test_logs
    log_success "Test logs saved to ${TEST_LOGS_DIR}/"
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline: preflight → build/pull → launch → shell
# ══════════════════════════════════════════════════════════════════════════════
TOTAL_STEPS=3
[[ "${MODE}" == "run" || "${MODE}" == "serve" ]] && TOTAL_STEPS=2

# ── [1/N] Pre-flight Checks ──────────────────────────────────────────────────
echo -e "\033[1;34m[1/${TOTAL_STEPS}] Pre-flight checks...\033[0m"

command -v docker &> /dev/null || log_error "docker not found."
if [[ ${#COMPOSE[@]} -eq 0 ]]; then
    log_error "Docker Compose not found."
fi

if ! docker info &> /dev/null; then
    log_error "Docker daemon not running or not accessible."
fi
log_success "Docker daemon running."

# ── [2/N] Build or Pull ──────────────────────────────────────────────────────
STEP=2
if [[ "${MODE}" == "build" ]]; then
    echo -e "\n\033[1;34m[${STEP}/${TOTAL_STEPS}] Synchronizing container image...\033[0m"
    "${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" \
        pull 2>/dev/null || log_warn "Pull failed or image is local-only. Assuming local image exists."
    log_success "Image synchronization complete."
    STEP=$((STEP + 1))
fi

# ── [N/N] Launch ──────────────────────────────────────────────────────────────
echo -e "\n\033[1;34m[${STEP}/${TOTAL_STEPS}] Launching container...\033[0m"

if [[ "${MODE}" == "serve" ]]; then
    log_info "Flushing host page caches and dentries to reclaim memory..."
    flush_page_cache
fi

docker stop "${WEDA_LLM_CONTAINER_NAME}" &> /dev/null || true
docker rm   "${WEDA_LLM_CONTAINER_NAME}" &> /dev/null || true
# checked after this container's own previous instance is gone; open-webui (3000) is this stack's own service, compose recreates it
REGISTRY_HOST="harbor.edgesync.cloud" weda_preflight "${WEDA_LLM_PORT:-9000}"

# ── Model cache pre-check ─────────────────────────────────────────────────────
# With HF_HUB_OFFLINE=1 (the .env default) the container refuses a model that is
# not in HF_CACHE; checking here names the download before anything is started.
if is_truthy "${HF_HUB_OFFLINE:-1}"; then
    if python3 -c 'import yaml' &> /dev/null; then
        CHECK_OUT="$(python3 "${SCRIPT_DIR}/scripts/download_models.py" --check 2>&1)" && CHECK_RC=0 || CHECK_RC=$?
        if [[ ${CHECK_RC} -ne 0 ]]; then
            echo "${CHECK_OUT}" | sed 's/^/  /' >&2 || true
            if echo "${CHECK_OUT}" | grep -q '^\[FAIL\]'; then
                log_error "The model to serve is not in HF_CACHE (${HF_CACHE}) -- download it or pick a cached profile (see above)."
            else
                log_error "model cache pre-check failed (see above)."
            fi
        fi
    else
        log_warn "python3 with pyyaml not found on the host -- model cache pre-check skipped (the container still refuses an uncached model)."
    fi
fi

# open-webui waits for a healthy API (compose depends_on). It is started after the
# wait below: a refused profile is then reported here at once, instead of compose
# sitting on "Waiting" for the whole health start period with no message.
"${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" \
    up -d --build --remove-orphans weda-llm

log_success "Container '${WEDA_LLM_CONTAINER_NAME}' is live."

# ── Wait for the model server ─────────────────────────────────────────────────
# The supervisor writes .ready while /health answers 200 and .launch-failed, with
# the reason, when it refuses the profile (port, cache, memory, bad config).
# TIMEOUT sizing: the active profile's own timeouts.server_ready_sec (read via
# scripts/download_models.py --ready-timeout, the same python3+pyyaml path the
# cache pre-check above uses) plus a 120s margin, floored at 1200s; falls back
# to the 1200s floor when pyyaml is not on the host.
TIMEOUT=1200
TIMEOUT_SOURCE="fallback default (pyyaml unavailable on host)"
if python3 -c 'import yaml' &> /dev/null; then
    if READY_SEC="$(python3 "${SCRIPT_DIR}/scripts/download_models.py" --ready-timeout 2>/dev/null)" && [[ "${READY_SEC}" =~ ^[0-9]+$ ]]; then
        TIMEOUT=$(( READY_SEC + 120 ))
        (( TIMEOUT < 1200 )) && TIMEOUT=1200
        TIMEOUT_SOURCE="active profile's server_ready_sec (${READY_SEC}s) + 120s margin"
    else
        TIMEOUT_SOURCE="fallback default (could not read server_ready_sec)"
    fi
fi
echo "  Waiting for the model server (TIMEOUT=${TIMEOUT}s, from ${TIMEOUT_SOURCE})..."
READY_FILE=/wise-edge/weda-llm-base/.ready
FAILED_FILE=/wise-edge/weda-llm-base/.launch-failed
elapsed=0
HYDRATION_OK=true
LAUNCH="pending"
while [[ $elapsed -lt $TIMEOUT ]]; do
    if docker exec "${WEDA_LLM_CONTAINER_NAME}" test -f "${READY_FILE}" 2>/dev/null; then
        printf "\r  [\033[1;32m%s\033[0m] 100%%\n" "$(printf "%*s" 50 "" | tr ' ' '#')"
        LAUNCH="ready"; break
    fi
    if docker exec "${WEDA_LLM_CONTAINER_NAME}" test -f "${FAILED_FILE}" 2>/dev/null; then
        echo -e "\n  [\033[1;31mREFUSED\033[0m]"
        LAUNCH="failed"; break
    fi
    if ! docker inspect "${WEDA_LLM_CONTAINER_NAME}" &> /dev/null; then
        echo -e "\n  [\033[1;31mMISSING\033[0m]"
        LAUNCH="missing"; break
    fi
    restarts="$(docker inspect -f '{{.RestartCount}}' "${WEDA_LLM_CONTAINER_NAME}" 2>/dev/null || echo 0)"
    if [[ "${restarts}" -ge 2 ]]; then
        echo -e "\n  [\033[1;31mCRASH LOOP\033[0m]"
        LAUNCH="crashloop"; break
    fi
    pct=$(( elapsed * 100 / TIMEOUT ))
    hashes=$(( (pct * 50) / 100 ))
    spaces=$(( 50 - hashes ))
    bar="$(printf "%*s" $hashes "" | tr ' ' '#')"
    empty="$(printf "%*s" $spaces "")"
    printf "\r  [\033[1;36m%s\033[0m%s] %3d%% " "$bar" "$empty" "$pct"
    sleep 2
    elapsed=$(( elapsed + 2 ))
done

case "${LAUNCH}" in
    ready) ;;
    failed)
        docker exec "${WEDA_LLM_CONTAINER_NAME}" cat "${FAILED_FILE}" 2>/dev/null | sed 's/^/  /' >&2 || true
        log_warn "The server refused to start (reason above). The container stays up idle: fix config/llm_config.yaml and save it to relaunch, or run ./build.sh --run again. Logs: docker logs ${WEDA_LLM_CONTAINER_NAME}"
        HYDRATION_OK=false ;;
    crashloop)
        log_warn "The server keeps crashing (${restarts} restarts). Logs: docker logs ${WEDA_LLM_CONTAINER_NAME}"
        HYDRATION_OK=false ;;
    missing)
        log_warn "Container '${WEDA_LLM_CONTAINER_NAME}' does not exist -- 'docker compose up' likely failed. Check: docker compose -f ${COMPOSE_FILE} --env-file ${ENV_FILE} logs weda-llm"
        HYDRATION_OK=false ;;
    *)
        echo -e "\n  [\033[1;31mTIMEOUT\033[0m]"
        log_warn "No .ready after ${TIMEOUT}s: the server may still be loading (large profiles take longer) or be stuck. Check: docker logs ${WEDA_LLM_CONTAINER_NAME}"
        HYDRATION_OK=false ;;
esac

# after a healthy API open-webui starts at once; after a timeout it is started
# without the dependency wait and connects when the API comes up
WEBUI_ARGS=()
[[ "${LAUNCH}" == "ready" ]] || WEBUI_ARGS=(--no-deps)
"${COMPOSE[@]}" -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" \
    up -d "${WEBUI_ARGS[@]}" open-webui \
    || log_warn "open-webui did not start; retry: docker compose up -d open-webui"

# ── Shell ─────────────────────────────────────────────────────────────────────
if [[ "${MODE}" == "serve" ]]; then
    log_success "Serving container launched in background."
    [[ "${HYDRATION_OK}" == "true" ]] || exit 1
    exit 0
fi

echo -e "\n\033[1;33m+══════════════════════════════════════════════════════════+\033[0m"
echo -e "\033[1;33m|       WEDA-LLM-BASE Pipeline — Shell                     |\033[0m"
echo -e "\033[1;33m+══════════════════════════════════════════════════════════+\033[0m"
echo -e "  Diagnostics:    \033[1;32m./scripts/wise-bench\033[0m"

docker exec -it "${WEDA_LLM_CONTAINER_NAME}" /bin/bash || true
# Shell is opened either way for debugging, but the script's own exit code
# must reflect whether hydration actually succeeded -- a silent 0 here would
# mask a crashed/never-ready container from anything scripting around this.
[[ "${HYDRATION_OK}" == "true" ]] || exit 1
