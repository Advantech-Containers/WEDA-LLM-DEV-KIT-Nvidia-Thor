# Troubleshooting Guide: WEDA-LLM-Cognitive-RAG serving on Jetson Thor

This guide covers common issues, root causes, and resolutions encountered when setting up, running, and benchmarking LLMs using the **Advantech WEDA-LLM-Cognitive-RAG Container** on the NVIDIA Jetson AGX Thor platform (`sm_110a` architecture).

---

## 📋 Table of Contents

1.  [First-Time Setup & Configuration](#1-first-time-setup--configuration)
    *   [Missing or Incomplete `.env` File](#missing-or-incomplete-env-file)
    *   [Script Permission Denied Errors](#script-permission-denied-errors)
    *   [Model Download Failures (`download_models.py`)](#model-download-failures-download_modelspy)
2.  [Container Build & Launch Issues](#2-container-build--launch-issues)
    *   [`./build.sh` Fails with "Docker daemon not running"](#buildsh-fails-with-docker-daemon-not-running)
    *   [Container Exits Immediately After Start](#container-exits-immediately-after-start)
    *   [Hydration Timeout During `./build.sh --run`](#hydration-timeout-during-buildsh---run)
    *   [No Module Named 'config' or 'polars' (ModuleNotFoundError)](#no-module-named-config-or-polars-modulenotfounderror)
3.  [Model Loading & vLLM Server Issues](#3-model-loading--vllm-server-issues)
    *   [vLLM Server Takes Too Long to Load the Model](#vllm-server-takes-too-long-to-load-the-model)
    *   [`/v1/models` Returns an Empty List](#v1models-returns-an-empty-list)
    *   [vLLM Crashes During Model Weight Loading](#vllm-crashes-during-model-weight-loading)
    *   [Hugging Face Cache and Path Mount Mismatches](#hugging-face-cache-and-path-mount-mismatches)
    *   [Model Hot-Swap Not Triggering After Editing `llm_config.yaml`](#model-hot-swap-not-triggering-after-editing-llm_configyaml)
4.  [Verification & Health Check Failures (`wise-bench`)](#4-verification--health-check-failures-wise-bench)
    *   [`wise-bench` Fails with "Container not running"](#wise-bench-fails-with-container-not-running)
    *   [Package Verification Suite Reports Failures](#package-verification-suite-reports-failures)
    *   [Endpoint Health Checks Fail (Open WebUI or vLLM)](#endpoint-health-checks-fail-open-webui-or-vllm)
    *   [Text Generation Test Returns Empty or Null Content](#text-generation-test-returns-empty-or-null-content)
5.  [Open WebUI Issues](#5-open-webui-issues)
    *   [Open WebUI Not Accessible on Port 3000](#open-webui-not-accessible-on-port-3000)
    *   [Open WebUI Shows "No Models Available"](#open-webui-shows-no-models-available)
6.  [API Endpoint & Inference Testing (`test_curl.sh`)](#6-api-endpoint--inference-testing-test_curlsh)
    *   [`test_curl.sh` Reports "No active model found"](#test_curlsh-reports-no-active-model-found)
    *   [Inference Returns Slow or Times Out](#inference-returns-slow-or-times-out)
7.  [JIT & Kernel Compilation Errors](#7-jit--kernel-compilation-errors)
    *   [ptxas-blackwell fatal : Value 'sm_110a' is not defined](#ptxas-blackwell-fatal--value-sm_110a-is-not-defined)
    *   [Cannot cache a graph with freezing enabled (RuntimeError)](#cannot-cache-a-graph-with-freezing-enabled-runtimeerror)
    *   [FlashInfer JIT Failures (NVRTC or curand_kernel.h errors)](#flashinfer-jit-failures-nvrtc-or-curand_kernelh-errors)
8.  [Model Accuracy & Quantization Bugs](#8-model-accuracy--quantization-bugs)
    *   [Garbage Output / Gibberish Tokens on Marlin Quantized MoE Models](#garbage-output--gibberish-tokens-on-marlin-quantized-moe-models)
    *   [Quantization method specified in the model config does not match](#quantization-method-specified-in-the-model-config-does-not-match)
9.  [Memory Management & Performance Deficits](#9-memory-management--performance-deficits)
    *   [Out of Memory (OOM) at Startup or "Free memory < budget"](#out-of-memory-oom-at-startup-or-free-memory--budget)
    *   [Reduced KV Cache Size or Unexpected Performance Degradation](#reduced-kv-cache-size-or-unexpected-performance-degradation)
    *   [Low Throughput or Inactive GPU Clocks](#low-throughput-or-inactive-gpu-clocks)
10. [Dependency & Maintenance Conflicts](#10-dependency--maintenance-conflicts)
    *   [Pip Prioritizing Generic x86_64 or Standard CPU Wheels](#pip-prioritizing-generic-x86_64-or-standard-cpu-wheels)
    *   [vLLM and Transformers Version Incompatibility](#vllm-and-transformers-version-incompatibility)
11. [Useful Links & Upstream Resources](#11-useful-links--upstream-resources)

---

## 1. First-Time Setup & Configuration

### Missing or Incomplete `.env` File
*   **Symptom:** `./build.sh` exits immediately with `Missing required vars in .env: WEDA_LLM_IMAGE WEDA_LLM_CONTAINER_NAME WEDA_LLM_PORT` or `.env not found`.
*   **Cause:** The `.env` file does not exist, or required variables are not set.
*   **Resolution:**
    1. Copy the template and fill in values:
       ```bash
       cp .env.example .env   # if an example exists, otherwise create manually
       ```
    2. Ensure the following variables are set at minimum:
       ```bash
       WEDA_LLM_IMAGE=harbor.edgesync.cloud/weda-ai/weda-llm-base:thor-1.0.0
       WEDA_LLM_CONTAINER_NAME=weda-edge-llm
       WEDA_LLM_PORT=9000
       HF_CACHE=/home/<username>/.cache/huggingface
       ```
    3. If using gated models (e.g. Llama, Gemma), set `HF_TOKEN` to a valid Hugging Face access token. An empty `HF_TOKEN` will emit a warning and fail on gated downloads.

### Script Permission Denied Errors
*   **Symptom:** Running `./build.sh`, `./wise-bench`, or `./test_curl.sh` fails with `Permission denied`.
*   **Cause:** The executable bit is not set on the script files.
*   **Resolution:**
    ```bash
    chmod +x build.sh entrypoint.sh wise-bench test_curl.sh scripts/wise-bench-llm scripts/serve.py scripts/download_models.py
    ```

### Model Download Failures (`download_models.py`)
*   **Symptom:** `python3 scripts/download_models.py` fails with authentication errors, network timeouts, or disk-space errors.
*   **Cause:** Invalid or missing `HF_TOKEN`, no network connectivity, or insufficient disk space on the cache volume.
*   **Resolution:**
    1. Verify your token: `huggingface-cli whoami` (should print your username).
    2. Ensure `HF_CACHE` in `.env` points to a volume with sufficient storage (large models like 70B+ require 40–140 GB).
    3. For air-gapped environments, set `HF_HUB_OFFLINE=1` in `.env` and pre-populate the cache from a machine with network access.

---

## 2. Container Build & Launch Issues

### `./build.sh` Fails with "Docker daemon not running"
*   **Symptom:** `./build.sh` or `./build.sh --run` aborts with `Docker daemon not running or not accessible` or `Cannot connect to the Docker daemon`.
*   **Cause:** Docker is not installed, the daemon is stopped, or the current user is not in the `docker` group.
*   **Resolution:**
    1. Start the Docker daemon: `sudo systemctl start docker`
    2. Add yourself to the docker group: `sudo usermod -aG docker $USER` (requires logout/login to take effect).
    3. Verify Docker + NVIDIA runtime:
       ```bash
       docker info | grep -i runtime
       ```
       You should see `nvidia` listed. If not, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and configure `/etc/docker/daemon.json`:
       ```json
       {
           "default-runtime": "nvidia",
           "runtimes": {
               "nvidia": {
                   "path": "nvidia-container-runtime",
                   "runtimeArgs": []
               }
           }
       }
       ```
    4. Restart Docker: `sudo systemctl restart docker`

### Container Exits Immediately After Start
*   **Symptom:** `docker ps` shows no running container after `./build.sh --run`. `docker ps -a` shows the container with status `Exited (1)`.
*   **Cause:** The entrypoint script encountered a fatal error (e.g. missing CUDA libraries, image mismatch, or corrupt volume mounts).
*   **Resolution:**
    1. Check exit logs:
       ```bash
       docker logs weda-edge-llm --tail 200
       ```
    2. Common causes and fixes:
       *   **"nvidia-smi: command not found"** — Container not launched with GPU access. Ensure `runtime: nvidia` is set in `docker-compose.yml` and `--gpus all` is passed.
       *   **"No CUDA detected"** — Verify JetPack SDK is installed: `dpkg -l | grep nvidia-l4t-core`
       *   **Triton ptxas patch failure** — The entrypoint prints a warning. Ensure `/usr/local/cuda/bin/ptxas` exists on the host and is mounted.
    3. For deeper debugging, start the container in shell mode:
       ```bash
       ./build.sh --shell
       ```
       If the container isn't running, temporarily change the compose command to `sleep infinity`, run `./build.sh --run`, then `docker exec -it weda-edge-llm bash`.

### Hydration Timeout During `./build.sh --run`
*   **Symptom:** `build.sh` polls for up to 1200s (20 minutes) and then prints `[TIMEOUT]`, `[REFUSED]`, or `[CRASH LOOP]` instead of reaching 100%.
*   **Cause:**
    *   **`[REFUSED]`:** `scripts/serve.py` gated the profile before launch -- the port is already in use, the model is missing from the offline cache, or the profile does not fit in memory -- and wrote the reason to `.launch-failed`.
    *   **`[CRASH LOOP]`:** the server started but died twice in a row (`RestartCount >= 2`); Docker keeps restarting it per `restart: unless-stopped`.
    *   **`[TIMEOUT]`:** `.ready` never appeared inside 1200s -- a large profile can still be loading (some declare `server_ready_sec` up to 3600s), or vLLM is stuck.
*   **Resolution:**
    1. `[REFUSED]` -- `build.sh` prints the reason (the contents of `.launch-failed`); fix `config/llm_config.yaml` for that reason and save it (the container idles and relaunches without a restart), or run `./build.sh --run` again.
    2. `[CRASH LOOP]` -- check `docker logs weda-edge-llm -f` for the vLLM traceback.
    3. `[TIMEOUT]` -- check `docker logs weda-edge-llm -f` to see whether the model is still loading. `open-webui` is started either way, with `--no-deps` after a timeout, and shows a connection error until the API answers healthy.
    4. In every case the container stays up (`docker ps`), so `./build.sh --shell` still works for diagnosis.

### No Module Named 'config' or 'polars' (ModuleNotFoundError)
*   **Symptom:** Starting the container or running the scoreboard checks outputs a traceback with `ModuleNotFoundError: No module named 'config'` or `No module named 'polars'`.
*   **Cause:**
    *   **config:** The `./config` folder on the host is not mounted to `/wise-edge/weda-llm-base/config` inside the container.
    *   **polars:** The `polars` Python library was not included in the base image.
*   **Resolution:**
    1. Check `docker-compose.yml` to verify that the config volume mount is set up correctly: `- ./config:/wise-edge/weda-llm-base/config:ro`.
    2. Ensure that you have stopped the container cleanly (`./build.sh --stop`) before rebuilding/re-running (`./build.sh --run`), as changes to volume definitions in docker-compose are not applied during a simple container restart.
    3. The container's `entrypoint.sh` automatically installs missing `polars` packages during startup if not present. Make sure you do not bypass the entrypoint using manual commands like `docker run ... bin/bash`.

---

## 3. Model Loading & vLLM Server Issues

### vLLM Server Takes Too Long to Load the Model
*   **Symptom:** After `./build.sh --run`, the container is running but `curl http://localhost:9000/v1/models` returns an error or empty response for an extended period.
*   **Cause:** Model weight loading time depends on model size, quantization format, and whether compilation caches are warm. First-time startup is always slower due to JIT compilation.
*   **Resolution:**
    1. Monitor loading progress in real time:
       ```bash
       docker logs weda-edge-llm -f
       ```
       Look for `[serve.py] Executing: vllm serve ...` followed by weight-loading progress.
    2. Expected approximate load times on Jetson Thor:

       | Model Size | Quantization | Typical Load Time |
       | :--- | :--- | :--- |
       | 0.5B–8B | BF16 / NVFP4 | 1–3 minutes |
       | 26B–30B (MoE) | NVFP4 / W4A16 | 3–8 minutes |
       | 70B+ | NVFP4 / BF16 | 10–25 minutes |
       | 120B (MoE) | MXFP4 | 15–30 minutes |

    3. First-run JIT compilation (Triton/Inductor) can add 1–5 minutes. Subsequent starts use the persistent `vllm-compile-cache` Docker volume and are significantly faster.
    4. If load time is excessive, check for OOM errors: `docker logs weda-edge-llm | grep -i "out of memory"`

### `/v1/models` Returns an Empty List
*   **Symptom:** `curl -s http://localhost:9000/v1/models` returns `{"data":[]}` or a connection refused error.
*   **Cause:** The vLLM server has not finished initialization, or it crashed during model loading.
*   **Resolution:**
    1. Verify the container is running: `docker ps | grep weda-edge-llm`
    2. Check if vLLM is still loading:
       ```bash
       docker logs weda-edge-llm --tail 50
       ```
       If you see active weight-loading or compilation messages, wait for completion.
    3. If the server died, the container restarts by itself (`docker ps` shows `Restarting` and the exit code); if the profile could not start at all, the container stays up and idle with the reason in the log -- fix `config/llm_config.yaml` and save it to relaunch. Check the specific error (OOM, missing model files, quantization mismatch) and refer to the relevant section of this guide.
    4. Retry the endpoint:
       ```bash
       watch -n 5 'curl -s http://localhost:9000/v1/models | python3 -m json.tool'
       ```

### vLLM Crashes During Model Weight Loading
*   **Symptom:** Container logs show a Python traceback during the weight-loading phase, and the container either exits or the vLLM process dies.
*   **Cause:** Common causes include: model checkpoint files are corrupt or incomplete, the model requires a `HF_TOKEN` that is not set, the specified `--revision` in `llm_config.yaml` does not exist, or there is a CUDA/driver version mismatch.
*   **Resolution:**
    1. **Corrupt or partial download:** Re-download the model:
       ```bash
       python3 scripts/download_models.py
       ```
       Then clear the compile cache and restart:
       ```bash
       docker volume rm weda-llm-vllm-compile-cache
       ./build.sh --stop && ./build.sh --run
       ```
    2. **Gated model without token:** Set `HF_TOKEN` in `.env` and ensure `HF_HUB_OFFLINE=0` if the model has not been previously cached.
    3. **Revision mismatch:** Check the `revision` field in `config/llm_config.yaml` and verify it exists on the Hugging Face repo.

### Hugging Face Cache and Path Mount Mismatches
*   **Symptom:** Serving script attempts to download model weights from the internet even though they have already been cached on the host, leading to timeouts in isolated/air-gapped networks.
*   **Cause:** The host model cache folder is not properly mapped to `/models/huggingface` inside the container, or the environment variable `HF_HUB_OFFLINE` is not set to `1`.
*   **Resolution:**
    1. Ensure the `HF_CACHE` variable in your `.env` points to the absolute directory on the host where Hugging Face downloads are stored (e.g. `HF_CACHE=/home/<username>/.cache/huggingface`).
    2. Validate that `docker-compose.yml` maps `${HF_CACHE}` to `/models/huggingface` (and maps tiktoken caches correctly if applicable).
    3. Set `HF_HUB_OFFLINE=1` in your `.env` or composer environment variables to force local cache verification and avoid external connection requests.

### Model Hot-Swap Not Triggering After Editing `llm_config.yaml`
*   **Symptom:** You edited `config/llm_config.yaml` to change the `default_profile` or model parameters, but the running server continues serving the old model.
*   **Cause:** The `scripts/serve.py` file watcher compares file modification times every 2 seconds. If the config file is mounted read-only (`:ro`), edits on the host should propagate, but some editors (e.g., vim with backup files, or editors that write-rename atomically) may not update the modification time as expected.
*   **Resolution:**
    1. Confirm the config is mounted: `docker exec weda-edge-llm ls -la /wise-edge/weda-llm-base/config/llm_config.yaml`
    2. Force a detectable change by touching the file after editing: `touch config/llm_config.yaml`
    3. Check container logs for the hot-swap message:
       ```bash
       docker logs weda-edge-llm --tail 20 | grep "hot-swap"
       ```
    4. If hot-swap is not working, restart the container manually:
       ```bash
       ./build.sh --stop && ./build.sh --run
       ```

---

## 4. Verification & Health Check Failures (`wise-bench`)

### `wise-bench` Fails with "Container not running"
*   **Symptom:** Running `./wise-bench` prints `Starting container first...` and then fails or hangs.
*   **Cause:** The `weda-edge-llm` container is not running. `wise-bench` attempts to start it via `./build.sh --serve`, which may not be a valid flag.
*   **Resolution:**
    1. Start the container first:
       ```bash
       ./build.sh --run
       ```
    2. Wait for model loading to complete (monitor with `watch -n 2 docker ps` and `docker logs weda-edge-llm -f`).
    3. Then run verification:
       ```bash
       ./wise-bench
       ```

### Package Verification Suite Reports Failures
*   **Symptom:** The "Container Package Verification Suite" section of `./wise-bench` output shows `[FAIL]` for one or more tiers (CUDA, imports, environment, compilation).
*   **Cause:** The container environment is missing dependencies, has version mismatches, or CUDA is not accessible.
*   **Resolution:**
    1. For **Tier-1 (CUDA capabilities)** failures: Verify NVIDIA runtime is configured and GPU is accessible inside the container:
       ```bash
       docker exec weda-edge-llm nvidia-smi
       docker exec weda-edge-llm python3 -c "import torch; print(torch.cuda.is_available())"
       ```
    2. For **Tier-2 (package import)** failures: A Python package is missing or broken. Check which import failed and install it:
       ```bash
       docker exec weda-edge-llm pip install <package-name> --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130
       ```
    3. For **Tier-3 (environment)** failures: Environment variables or volume mounts are misconfigured. Compare your `.env` and `docker-compose.yml` against the expected values.
    4. For **Tier-4 (compilation)** failures: JIT/Triton kernel compilation is broken. See [JIT & Kernel Compilation Errors](#7-jit--kernel-compilation-errors).
    5. Run the full scorecard for a detailed report:
       ```bash
       ./build.sh --test
       ```

### Endpoint Health Checks Fail (Open WebUI or vLLM)
*   **Symptom:** The `wise-bench` endpoint checks show `[FAIL]` for Open WebUI (port 3000) or vLLM API (port 9000).
*   **Cause:** The respective service is not running, not yet initialized, or the port is blocked.
*   **Resolution:**
    1. Check which services are running:
       ```bash
       docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
       ```
    2. For **Open WebUI** failures: See [Open WebUI Not Accessible on Port 3000](#open-webui-not-accessible-on-port-3000).
    3. For **vLLM API** failures: The model may still be loading. See [vLLM Server Takes Too Long to Load the Model](#vllm-server-takes-too-long-to-load-the-model).
    4. Check for port conflicts: `sudo lsof -i :9000` and `sudo lsof -i :3000`

### Text Generation Test Returns Empty or Null Content
*   **Symptom:** The `wise-bench` text generation test shows `[FAIL] Completion check failed, returned empty, or returned null content`.
*   **Cause:** The vLLM server is running but the model produced no output (possible quantization issue, empty response, or reasoning-only model).
*   **Resolution:**
    1. Test manually with a simple prompt:
       ```bash
       curl -s -X POST http://localhost:9000/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{"model": "<your-model-id>", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 64}'
       ```
    2. If the response has `reasoning` but no `content`, the model may be a reasoning-only variant. This is expected behavior for some models.
    3. If the response is truly empty, the model may be producing garbage. See [Garbage Output / Gibberish Tokens on Marlin Quantized MoE Models](#garbage-output--gibberish-tokens-on-marlin-quantized-moe-models).

---

## 5. Open WebUI Issues

### Open WebUI Not Accessible on Port 3000
*   **Symptom:** Navigating to `http://localhost:3000` in a browser returns "connection refused" or a blank page.
*   **Cause:** The `open-webui` container is not running, is still initializing, or port 3000 is occupied by another service.
*   **Resolution:**
    1. Check the Open WebUI container status:
       ```bash
       docker ps | grep open-webui
       docker logs open-webui --tail 50
       ```
    2. If not running, ensure both services were launched:
       ```bash
       ./build.sh --stop && ./build.sh --run
       ```
       The `build.sh` script starts both `weda-llm` and `open-webui` services.
    3. If port 3000 is in use by another application, stop the conflicting service or change the `PORT` environment variable for Open WebUI in `docker-compose.yml`.

### Open WebUI Shows "No Models Available"
*   **Symptom:** Open WebUI loads in the browser but the model selector dropdown is empty.
*   **Cause:** Open WebUI connects to vLLM at `http://127.0.0.1:<WEDA_LLM_PORT>/v1`, and the vLLM server has not finished loading the model yet.
*   **Resolution:**
    1. Wait for vLLM to finish loading:
       ```bash
       watch -n 5 'curl -s http://localhost:9000/v1/models | python3 -m json.tool'
       ```
    2. Once the models endpoint returns data, refresh the Open WebUI page.
    3. If the issue persists, verify the connection URL in `docker-compose.yml`:
       ```yaml
        - OPENAI_API_BASE_URL=http://127.0.0.1:${WEDA_LLM_PORT:-9000}/v1
       ```

---

## 6. API Endpoint & Inference Testing (`test_curl.sh`)

### `test_curl.sh` Reports "No active model found"
*   **Symptom:** Running `./test_curl.sh` prints `[ERROR] No active model found. Ensure the serving container is running.`
*   **Cause:** The vLLM `/v1/models` endpoint returned no models — either the server is still loading or has crashed.
*   **Resolution:**
    1. Check vLLM status:
       ```bash
       curl -s http://localhost:9000/v1/models | python3 -m json.tool
       ```
    2. If empty, wait for the model to load. Monitor with: `docker logs weda-edge-llm -f`
    3. If the container is not running: `./build.sh --run`

### Inference Returns Slow or Times Out
*   **Symptom:** `test_curl.sh` or manual curl requests take minutes to return, or curl times out entirely.
*   **Cause:** The model has just loaded and is performing first-request JIT compilation, the GPU clocks are throttled, or the `max_tokens` is set very high.
*   **Resolution:**
    1. The first request after server startup is always slower due to warm-up compilation. Run a second request to measure true performance.
    2. Lock GPU clocks to max performance:
       ```bash
       sudo nvpmodel -m 0 && sudo jetson_clocks
       ```
    3. For quick tests, reduce `max_tokens` in your curl command (e.g., 64 instead of 1024).

---

## 7. JIT & Kernel Compilation Errors

### ptxas-blackwell fatal : Value 'sm_110a' is not defined
*   **Symptom:** During compilation of custom kernels, PyTorch Inductor or Triton throws a fatal compilation error: `ptxas-blackwell fatal : Value 'sm_110a' is not defined for option 'gpu-name'`.
*   **Cause:** Triton distributes its own bundled versions of the `ptxas` assembler which predate Blackwell and are not aware of the Jetson Thor `sm_110a` target architecture.
*   **Resolution:**
    *   The container's `entrypoint.sh` includes an automatic routing patch. It scans for the system's native Blackwell-aware `ptxas` (from CUDA 13.0+) at `/usr/local/cuda/bin/ptxas` and symlinks it over Triton's bundled `ptxas` binary.
    *   If launching without the entrypoint, you can manually force Triton to use the correct binary by setting the environment variables:
        ```bash
        export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
        export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
        ```
    *   For detail on this Triton compilation logic, see the [Triton Issue Tracker](https://github.com/triton-lang/triton/issues).

### Cannot cache a graph with freezing enabled (RuntimeError)
*   **Symptom:** Server execution fails with `RuntimeError: Cannot cache a graph with freezing enabled` or `Unsupported node from codegen` during the graph capture phase.
*   **Cause:** A known regression in PyTorch Inductor where manually enabling "freezing" optimizations (which bakes parameters as constants) conflicts with vLLM's internal compiler graph caching.
*   **Resolution:**
    1. Set the Inductor freezing configuration to `False` to bypass the collision.
    2. Add or verify the compile arguments in your model serving command (or within `config/llm_config.yaml`):
       `--compilation-config='{"inductor_compile_config":{"freezing":false}}'`
    3. Ensure `TORCHINDUCTOR_FREEZING=0` is exported in the container launch environment. Refer to the [PyTorch Inductor Documentation](https://pytorch.org/docs/stable/torch.compiler.html) for detailed parameters on graph capturing.

### FlashInfer JIT Failures (NVRTC or curand_kernel.h errors)
*   **Symptom:** Models utilizing FlashInfer fail to initialize with tracebacks detailing JIT compilation errors related to missing development header files (e.g. `curand_kernel.h`, `nvrtc.h` not found).
*   **Cause:** Minimal container environments remove heavy development headers to keep image sizes small, preventing dynamic compiling of JIT kernels.
*   **Resolution:**
    *   Route around the JIT compilation step entirely. In `entrypoint.sh` or through your container environment variables, enforce the use of CUTLASS for FlashInfer backends:
        ```bash
        VLLM_USE_FLASHINFER_MOE_FP4=0
        VLLM_NVFP4_GEMM_BACKEND=cutlass
        ```
    *   This forces the server to use precompiled, Sm110a-compatible binary kernels rather than compiling them dynamically on startup.

---

## 8. Model Accuracy & Quantization Bugs

### Garbage Output / Gibberish Tokens on Marlin Quantized MoE Models
*   **Symptom:** The served model returns repetitive punctuation (e.g. repeating `!`), gibberish text, or highly erratic tokens under normal inference.
*   **Cause:** The Marlin quantized `W4A16` (or `compressed-tensors`) Mixture-of-Experts (MoE) dequantization kernel contains arithmetic incompatibilities with the Blackwell `sm_110a` instruction set.
*   **Resolution:**
    *   Do not run `W4A16` quantized MoE models on Jetson Thor.
    *   To maintain high performance and compatibility, run native `NVFP4` (Quantized FP4), `FP8`, or standard `BF16` precision formats instead. For active discussions, consult the [vLLM Project Issue Tracker](https://github.com/vllm-project/vllm/issues).

### Quantization method specified in the model config does not match
*   **Symptom:** Server crashes immediately on startup with a mismatch error regarding the quantization configuration.
*   **Cause:** Manually forcing the quantization flag (e.g. `--quantization=awq_marlin`) on a model config that requires a different layout (e.g. `compressed-tensors`).
*   **Resolution:**
    *   Omit the `--quantization` flag entirely from the command parameters and allow vLLM to automatically read and map the correct quantization method defined in the model's own `config.json`.

---

## 9. Memory Management & Performance Deficits

### Out of Memory (OOM) at Startup or "Free memory < budget"
*   **Symptom:** Serving container aborts startup during the KV cache allocation phase with an Out of Memory error.
*   **Cause:** The configured `gpu_memory_utilization` is too high, or previously terminated runs have left allocated pages in the unified memory (UMA) cache.
*   **Resolution:**
    1. Terminate all duplicate containers and clean page caches on the host before launching a new model:
       ```bash
       docker ps -q | xargs -r docker rm -f
       sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
       ```
    2. If the OOM persists, lower the `gpu_memory_utilization` setting (e.g., from `0.90` to `0.85` or `0.80`) in the model's profile inside `config/llm_config.yaml`.

### Reduced KV Cache Size or Unexpected Performance Degradation
*   **Symptom:** Concurrency tests fail at lower thresholds than normal, and logs indicate a low total number of allocated GPU KV cache blocks.
*   **Cause:** Fragmented system memory page caches are consuming critical unified memory, preventing vLLM from reserving the configured memory budget.
*   **Resolution:**
    *   Flush host page caches and dentries prior to model load using:
        ```bash
        sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
        ```
    *   `scripts/serve.py` automatically attempts to drop system page caches before starting vLLM to ensure maximum contiguous memory availability.

### Low Throughput or Inactive GPU Clocks
*   **Symptom:** Inference speeds (tokens/sec) are significantly lower than established benchmarks.
*   **Cause:** The Jetson CPU/GPU clocks are running in low-power throttling states, or the concurrency test is too low to saturate the GPU.
*   **Resolution:**
    1. Lock system clocks to the maximum profile before benchmarking:
       ```bash
       sudo nvpmodel -m 0 && sudo jetson_clocks
       ```
    2. When utilizing the benchmark suite, specify the clock lock flag:
       `python3 scripts/wise-bench-llm --lock-clocks`
    3. Ensure you sweep concurrency (`--concurrency-sweep "8,16,32,64,128"`) to find the true saturation point of the platform.

---

## 10. Dependency & Maintenance Conflicts

### Pip Prioritizing Generic x86_64 or Standard CPU Wheels
*   **Symptom:** Installing packages inside the container pulls CPU-only versions of torch or packages that lack CUDA capability.
*   **Cause:** PyPI's default indexing prioritizes generic wheels over Jetson-specific wheels when resolving dependencies.
*   **Resolution:**
    *   Always point pip to the official Jetson AI Lab packages index when installing or upgrading dependencies:
        ```bash
        pip install --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130 <package-name>
        ```
    *   For detail on available builds, consult the [Jetson AI Lab Software Guide](https://www.jetson-ai-lab.com/).

### vLLM and Transformers Version Incompatibility
*   **Symptom:** vLLM crashes with initialization errors referencing missing or changed APIs inside the `transformers` library.
*   **Cause:** A version collision where vLLM is compiled against a specific major/minor release, but a newer `transformers` release is present in the python path.
*   **Resolution:**
    *   Pin library versions strictly in your environment setup. For example, vLLM 0.20 requires pinning `transformers==5.6.0` to avoid code path breakages.

---

## 11. Useful Links & Upstream Resources

*   **vLLM Project Repository:** [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
*   **vLLM Official Documentation:** [docs.vllm.ai](https://docs.vllm.ai/en/latest/)
*   **Triton Lang Compiler Issues:** [github.com/triton-lang/triton/issues](https://github.com/triton-lang/triton/issues)
*   **NVIDIA Jetson AI Lab Guides:** [jetson-ai-lab.com](https://www.jetson-ai-lab.com/)
*   **PyTorch Compiler Documentation:** [pytorch.org/docs/stable/torch.compiler.html](https://pytorch.org/docs/stable/torch.compiler.html)
*   **NVIDIA Container Toolkit Repository:** [github.com/NVIDIA/nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
*   **Hugging Face Hub Documentation:** [huggingface.co/docs/hub](https://huggingface.co/docs/hub)
*   **Open WebUI Documentation:** [docs.openwebui.com](https://docs.openwebui.com/)
