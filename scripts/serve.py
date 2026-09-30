#!/usr/bin/env python3
# ==============================================================================
# WEDA-LLM-BASE — Model Server Supervisor
#
# Launches vLLM from the active profile in config/llm_config.yaml and runs as
# the container's main process under docker-init (docker-compose sets init:
# true, so this is not PID 1). Contract:
#   * a running server that dies is a container that exits with the server's code, so
#     compose (restart: unless-stopped) restarts it and `docker ps` tells the
#     truth -- never an "Up" container with no server behind it;
#   * the one exception is a crash right after a hot swap (a bad new profile):
#     the supervisor idles and waits for the next config save instead of
#     taking a working container down;
#   * /wise-edge/weda-llm-base/.ready exists only while /health answers 200, and
#     .launch-failed holds the reason while the supervisor idles, so a launcher
#     stops waiting the moment a profile is refused instead of at its timeout;
#   * SIGTERM stops the server's whole process group and exits 0;
#   * a profile whose gpu_memory_utilization cannot be satisfied by the memory
#     available right now, a model absent from the offline cache, or a port that
#     another process already serves are refused before vLLM allocates anything
#     (the container idles with the reason instead of crash-looping).
# ==============================================================================
import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Ensure the root directory is in sys.path so config.loader can be found natively
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import load_llm_config

READY_FILE = "/wise-edge/weda-llm-base/.ready"
FAILED_FILE = "/wise-edge/weda-llm-base/.launch-failed"
DEFAULT_READY_TIMEOUT = 1200
SWAP_GRACE_SEC = 120      # a crash this soon after a hot swap is the new config's fault
MEMORY_MARGIN_GIB = 1.0   # headroom kept for the server process itself


def env_first(*names, default=None):
    """First set environment variable among `names` (bundle naming first, then
    the knobs baked into the image)."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def _is_offline(default):
    """Same truthy set download_models.py's pre-check uses, case-insensitively."""
    return os.environ.get("HF_HUB_OFFLINE", default).strip().lower() in ("1", "true", "yes")


def meminfo_gib():
    """(MemTotal, MemAvailable) in GiB from /proc/meminfo, or (None, None)."""
    try:
        vals = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                vals[k] = int(rest.split()[0])
        return vals["MemTotal"] / 2**20, vals["MemAvailable"] / 2**20
    except Exception:
        return None, None


def fit_gate(profile):
    """Refuse a profile whose vLLM reservation exceeds what is free right now.
    vLLM's gpu_memory_utilization is a fraction of TOTAL memory on a Jetson
    (unified memory); the reservation must be actually free at startup or
    the engine dies after minutes of loading."""
    total, avail = meminfo_gib()
    if total is None:
        print("[serve] WARN: /proc/meminfo unreadable -- skipping the memory fit check", flush=True)
        return
    need = float(profile.gpu_memory_utilization) * total + MEMORY_MARGIN_GIB
    if need > avail:
        raise RuntimeError(
            f"profile '{profile.served_model_name}' does not fit: gpu_memory_utilization "
            f"{profile.gpu_memory_utilization} of {total:.1f} GiB = {need:.1f} GiB needed "
            f"(incl. {MEMORY_MARGIN_GIB:.0f} GiB margin) vs {avail:.1f} GiB available. "
            "Free memory (stop other workloads) or lower gpu_memory_utilization.")
    print(f"[serve] memory fit check: {need:.1f} GiB needed, {avail:.1f} GiB available -- OK", flush=True)


def cached_snapshot(repo_id, cache_root, revision=None):
    """True when the HF hub cache under `cache_root` holds a snapshot of
    `repo_id` with a config.json (what vLLM needs to start offline). When
    `revision` is a pinned commit sha, the snapshot dir name must start with
    it -- a cache holding a different revision is not a match."""
    import glob
    if os.path.isdir(repo_id):
        return True
    name = "models--" + repo_id.replace("/", "--")
    for hub in (os.path.join(cache_root, "hub"), cache_root):
        for path in glob.glob(os.path.join(hub, name, "snapshots", "*", "config.json")):
            snap = os.path.basename(os.path.dirname(path))
            if revision is None or snap.startswith(revision):
                return True
    return False


def port_gate(host, port):
    """Refuse to launch onto a port something else already serves (host network:
    the bind seen here is the host's). A crash-restart loop would hide the cause;
    idling with the holder's name does not."""
    import socket
    bind_host = host if host not in ("0.0.0.0", "") else "0.0.0.0"
    try:
        family = socket.getaddrinfo(bind_host, None)[0][0]
    except socket.gaierror as e:
        raise RuntimeError(f"cannot resolve bind address {host}: {e}")

    # right after a hot swap the previous server's socket may still be closing
    # (TIME_WAIT); a fresh bind attempt a couple seconds later clears that up
    # without refusing a profile the previous one just vacated.
    last_err = None
    for attempt in range(3):
        probe = socket.socket(family, socket.SOCK_STREAM)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((bind_host, int(port)))
            return
        except OSError as e:
            last_err = e
            if attempt < 2:
                time.sleep(2)
                continue
        finally:
            probe.close()

    who = " (held by a process outside this container (see ss -ltnp on the host))"
    try:
        out = subprocess.run(["sh", "-c", f"ss -Hltnp 2>/dev/null | awk '$4 ~ /[:.]{int(port)}$/ {{print $NF; exit}}'"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if out and "users:(" in out:
            who = f" (held by {out})"
    except Exception:
        pass
    raise RuntimeError(f"port {port} on {host} is already in use{who}: stop that process or set another port in .env ({last_err.strerror})")


def offline_gate(profile):
    """With HF_HUB_OFFLINE set, a model that is not in the mounted cache would
    make vLLM die minutes later; refuse it up front with the fix spelled out."""
    if not _is_offline("0"):
        return
    root = profile.model_cache_dir or os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME") or "/models/huggingface"
    if not cached_snapshot(profile.model, root, profile.revision):
        raise RuntimeError(
            f"model '{profile.model}' is not in the cache mounted at {root} and HF_HUB_OFFLINE is set. "
            "On the host run: python3 scripts/download_models.py --profile <name> (or set HF_HUB_OFFLINE=0 in .env).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default=None, help="Config profile to use")
    args = parser.parse_args()

    config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "llm_config.yaml")
    # knob names: LLM_* (the weda-llm-base bundle repo's release naming), WEDA_LLM_*
    # (the application demos), L3A_LLM_* (baked into the image) -- first one set wins
    host = env_first("LLM_HOST", "WEDA_LLM_HOST", "L3A_LLM_HOST", default="0.0.0.0")
    port = env_first("LLM_PORT", "WEDA_LLM_PORT", "L3A_LLM_PORT", default="9000")

    def build_cmd(profile):
        cmd = [
            "vllm", "serve", profile.model,
            "--host", host,
            "--port", port,
            "--served-model-name", profile.served_model_name,
            "--max-model-len", str(profile.max_model_len),
            "--max-num-seqs", str(profile.max_num_seqs),
            "--gpu-memory-utilization", str(profile.gpu_memory_utilization),
            "--tensor-parallel-size", str(profile.tensor_parallel_size),
        ]
        if profile.trust_remote_code:
            cmd.append("--trust-remote-code")
        if profile.revision:
            cmd.extend(["--revision", profile.revision])
        if profile.model_cache_dir:
            cmd.extend(["--download-dir", profile.model_cache_dir])
        if profile.extra_serve_args:
            cmd.extend(profile.extra_serve_args)
        return cmd

    def build_env(profile):
        env = os.environ.copy()
        env["TORCHINDUCTOR_FREEZING"] = str(profile.inductor_freezing)
        env["LLM_INDUCTOR_FREEZING"] = str(profile.inductor_freezing)
        env["WEDA_LLM_INDUCTOR_FREEZING"] = str(profile.inductor_freezing)
        env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = str(profile.fx_graph_cache)
        env["VLLM_USE_FLASHINFER_MOE_FP4"] = str(profile.use_flashinfer_moe_fp4)
        env["VLLM_NVFP4_GEMM_BACKEND"] = profile.nvfp4_gemm_backend
        for k, v in (profile.env or {}).items():
            env[k] = str(v)
        return env

    def launch():
        """Load the profile, gate it, start vLLM. Raises on a bad profile."""
        profile = load_llm_config(args.profile, config_path=config_file)
        if profile.model_cache_dir:
            os.environ["HUGGINGFACE_HUB_CACHE"] = profile.model_cache_dir
        print(f"[serve] Loading profile: {args.profile or 'default'} -> {profile.served_model_name}")
        print(f"[serve] Model path: {profile.model}")
        port_gate(host, port)
        offline_gate(profile)
        try:
            subprocess.run(["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True,
                           capture_output=True, timeout=30)
            print("[serve] page caches flushed to reclaim unified memory")
        except Exception as e:
            print(f"[serve] WARN: could not flush caches: {e}")
        # MemAvailable after the flush is the closest non-CUDA proxy for the free-memory
        # snapshot vLLM's own gpu_memory_utilization check takes at init_device time.
        fit_gate(profile)
        cmd = build_cmd(profile)
        print(f"[serve] Executing: {' '.join(cmd)}", flush=True)
        timeout = (profile.timeouts or {}).get("server_ready_sec", DEFAULT_READY_TIMEOUT)
        return subprocess.Popen(cmd, env=build_env(profile), start_new_session=True), timeout

    state = {"proc": None, "ready_deadline": None, "ready": False, "swapped_at": None, "stopping": False}

    def idle(reason):
        print(f"\n[serve] SERVER DOWN: {reason}\n"
              "[serve] idle -- fix config/llm_config.yaml and save it (or touch it) to relaunch;"
              " the container stays up\n", file=sys.stderr, flush=True)
        state["ready_deadline"], state["ready"] = None, False
        mark_failed(reason)
        return subprocess.Popen(["sleep", "infinity"], start_new_session=True), None

    def healthy():
        try:
            return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200
        except Exception:
            return False

    def mark_ready():
        os.makedirs(os.path.dirname(READY_FILE), exist_ok=True)
        with open(READY_FILE, "w"):
            pass

    def clear_ready():
        try:
            os.remove(READY_FILE)
        except FileNotFoundError:
            pass

    def mark_failed(reason):
        try:
            os.makedirs(os.path.dirname(FAILED_FILE), exist_ok=True)
            with open(FAILED_FILE, "w") as f:
                f.write(str(reason).rstrip() + "\n")
        except OSError as e:
            print(f"[serve] WARN: could not write {FAILED_FILE}: {e}", file=sys.stderr, flush=True)

    def clear_failed():
        try:
            os.remove(FAILED_FILE)
        except FileNotFoundError:
            pass

    def stop(proc, timeout=30):
        if not (proc and proc.poll() is None):
            return
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=timeout)
        except Exception:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    # mtime is captured BEFORE the first launch so an edit made during a
    # multi-minute cold load is still picked up
    last_mtime = os.path.getmtime(config_file)
    clear_ready()
    clear_failed()

    def start(label):
        clear_failed()
        try:
            proc, timeout = launch()
            state["ready_deadline"] = time.time() + timeout
            print(f"[serve] waiting up to {timeout}s for {label} to report healthy...", flush=True)
        except (SystemExit, Exception) as e:
            if state["stopping"]:
                # SIGTERM arrived while launch() was running: on_term's exit
                # was caught above as a SystemExit -- this is a shutdown, not
                # a bad profile, so no .launch-failed and no idle() sleeper
                proc = None
            else:
                proc, _ = idle(f"{label} failed to launch: {e}")
        state["proc"], state["ready"] = proc, False
        if state["stopping"]:
            stop(state["proc"])
            sys.exit(0)

    def on_term(signum, frame):
        state["stopping"] = True
        clear_ready()
        stop(state["proc"])
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    start("initial launch")
    # a profile that cannot launch (bad config, does not fit) leaves the
    # container up and idle with the reason in the log: fix the profile and
    # save it; a restart loop would hide the message

    try:
        while True:
            time.sleep(2)
            proc = state["proc"]
            rc = proc.poll()
            if rc is not None and not state["stopping"]:
                clear_ready()
                if state["swapped_at"] and time.time() - state["swapped_at"] < SWAP_GRACE_SEC:
                    state["proc"], _ = idle(f"the new server exited with code {rc} (see the vLLM lines above)")
                    state["swapped_at"] = None
                    continue
                print(f"[serve] vLLM exited with code {rc} -- exiting so the container restarts", file=sys.stderr, flush=True)
                sys.exit(rc)
            if not state["ready"] and state["ready_deadline"] is not None:
                if healthy():
                    mark_ready()
                    state["ready"] = True
                    print("[serve] server is healthy -- marked ready", flush=True)
                elif time.time() > state["ready_deadline"]:
                    print("[serve] WARN: server still not healthy after its server_ready_sec budget "
                          "-- it keeps running; check the vLLM lines above", file=sys.stderr, flush=True)
                    state["ready_deadline"] = None
            try:
                mtime = os.path.getmtime(config_file)
            except OSError:
                continue
            if mtime != last_mtime:
                last_mtime = mtime
                print("\n[serve] llm_config.yaml changed -- hot-swapping the model (container stays up)\n", flush=True)
                clear_ready()
                stop(proc)
                start("hot-swapped server")
                state["swapped_at"] = time.time()
    except KeyboardInterrupt:
        clear_ready()
        stop(state["proc"])


if __name__ == "__main__":
    main()
