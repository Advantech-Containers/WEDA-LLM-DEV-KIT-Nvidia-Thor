#!/usr/bin/env python3
# Copyright (c) 2026 Advantech Co., Ltd.
# Script: download_models.py
# Downloads and caches all models defined in the benchmarking profiles.

import argparse
import os
import sys
import pathlib

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None   # only downloading needs it; --list and --check run without

try:
    import yaml
except ImportError:
    print("[ERROR] pyyaml is not installed on the host.")
    print("Please install it using: pip install pyyaml")
    sys.exit(1)

CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "llm_config.yaml"

ENV_PATH = pathlib.Path(__file__).resolve().parent.parent / ".env"


def _load_dotenv(path=ENV_PATH):
    """Populate os.environ from .env (KEY=VALUE per line, ${VAR} expanded like
    the shell and compose do). Real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), os.path.expandvars(value.strip().strip('"').strip("'")))



def _load_config(config_path=CONFIG_PATH):
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if not (cfg.get("profiles") or {}):
        print(f"[ERROR] no profiles found in {config_path}", file=sys.stderr)
        sys.exit(1)
    return cfg


def load_models(config_path=CONFIG_PATH, profile=None, only_default=False):
    """Models to fetch, always resolved THROUGH config/llm_config.yaml.

    No selection path ever names a repo id directly, so a profile's model is never
    missed here as the roster grows.

      default          every model referenced by any profile
      only_default     just the model served by default_profile
      profile=<name>   just that profile's model
    """
    cfg = _load_config(config_path)
    profiles = cfg["profiles"]

    if only_default:
        profile = cfg.get("default_profile")
        if not profile:
            print(f"[ERROR] no default_profile set in {config_path}", file=sys.stderr)
            sys.exit(1)

    if profile:
        if profile not in profiles:
            print(f"[ERROR] profile '{profile}' not found in {config_path}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(profiles))}", file=sys.stderr)
            sys.exit(1)
        print(f"profile: {profile} -> {profiles[profile]['model']}")
        return [profiles[profile]["model"]]

    models = sorted({p["model"] for p in profiles.values() if (p or {}).get("model")})
    default = cfg.get("default_profile")
    if default and default in profiles:
        print(f"default_profile: {default} -> {profiles[default]['model']}")
    return models


def cached_snapshot(repo_id, hf_home, revision=None):
    """True when the hub cache under `hf_home` holds a snapshot of `repo_id`
    with a config.json -- the same test the container's supervisor applies
    before it launches, so a refusal is reported here, before anything starts.
    When `revision` is a pinned commit sha, the snapshot dir name must start
    with it -- a cache holding a different revision is not a match."""
    import glob
    if os.path.isdir(repo_id):
        return True
    name = "models--" + repo_id.replace("/", "--")
    for hub in (os.path.join(hf_home, "hub"), hf_home):
        for path in glob.glob(os.path.join(hub, name, "snapshots", "*", "config.json")):
            snap = os.path.basename(os.path.dirname(path))
            if revision is None or snap.startswith(revision):
                return True
    return False


def list_profiles(config_path=CONFIG_PATH):
    cfg = _load_config(config_path)
    default = cfg.get("default_profile")
    for name in sorted(cfg["profiles"]):
        mark = " (default)" if name == default else ""
        print(f"  {name}{mark}\n      {cfg['profiles'][name].get('model')}")


# Mirrors serve.py's DEFAULT_READY_TIMEOUT -- used only when a profile declares
# no timeouts.server_ready_sec of its own.
DEFAULT_READY_TIMEOUT = 1200


def ready_timeout(config_path=CONFIG_PATH, profile=None):
    """The named (or default_profile's) timeouts.server_ready_sec, for build.sh's
    TIMEOUT calculation -- read the same way the cache pre-check reads the config."""
    cfg = _load_config(config_path)
    profiles = cfg["profiles"]
    if not profile:
        profile = cfg.get("default_profile")
    p = profiles.get(profile) or {}
    return int((p.get("timeouts") or {}).get("server_ready_sec", DEFAULT_READY_TIMEOUT))


def parse_args():
    ap = argparse.ArgumentParser(
        description="Download models referenced by config/llm_config.yaml.")
    g = ap.add_mutually_exclusive_group()
    # No flag downloads ONLY the default profile's model. Fetching all 12 -- which
    # includes a 70B and a 120B -- must be asked for explicitly, so a forgotten
    # flag costs nothing instead of hundreds of GB.
    g.add_argument("--all", action="store_true",
                   help="download every model referenced by any profile")
    g.add_argument("--profile", metavar="NAME",
                   help="download only this profile's model")
    g.add_argument("--list", action="store_true",
                   help="list available profiles and their models, download nothing")
    ap.add_argument("--check", action="store_true",
                    help="download nothing: exit 0 when the default profile's model (or "
                         "--profile NAME's) is already in the cache, 1 with the download "
                         "command otherwise; build.sh runs this before launching")
    ap.add_argument("--ready-timeout", action="store_true",
                    help="print the active (or --profile NAME's) profile's "
                         "timeouts.server_ready_sec and exit; used by build.sh to size its wait")
    return ap.parse_args()


def _is_offline(default):
    """Same truthy set serve.py's offline_gate uses, so a value that disables
    the gate in the container also skips the pre-check here."""
    return os.environ.get("HF_HUB_OFFLINE", default).strip().lower() in ("1", "true", "yes")


def _profile_revisions(config_path, profile=None, all_profiles=False):
    """model -> pinned revision (or None) for whichever profiles are in scope,
    so a cache holding the wrong revision of an otherwise-cached model is
    still reported as missing."""
    cfg = _load_config(config_path)
    profiles = cfg["profiles"]
    if all_profiles:
        scoped = profiles
    elif profile:
        scoped = {profile: profiles[profile]} if profile in profiles else {}
    else:
        default = cfg.get("default_profile")
        scoped = {default: profiles[default]} if default in profiles else {}
    revisions = {}
    for p in scoped.values():
        model = (p or {}).get("model")
        if model:
            revisions.setdefault(model, (p or {}).get("revision"))
    return revisions


def check_cached(profile=None, all_profiles=False, config_path=CONFIG_PATH):
    """0 when the model(s) to be served are in the cache, 1 (with the fix) when not."""
    hf_home = os.path.expandvars(os.environ.get("HF_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    if not _is_offline("1"):
        print("HF_HUB_OFFLINE is off: the server fetches a missing model at start-up; nothing to pre-check.")
        return 0
    models = load_models(config_path=config_path, profile=profile, only_default=not (profile or all_profiles))
    revisions = _profile_revisions(config_path, profile=profile, all_profiles=all_profiles)
    missing = []
    for m in models:
        if os.path.isabs(m):
            print(f"cannot pre-check a local path {m} from the host; the container checks it")
            continue
        if not cached_snapshot(m, hf_home, revisions.get(m)):
            rev = revisions.get(m)
            missing.append(f"{m} (revision {rev})" if rev else m)
    if missing:
        print(f"[FAIL] not in the cache under {os.path.join(hf_home, 'hub')}: {', '.join(missing)}", file=sys.stderr)
        print("       download it first:  python3 scripts/download_models.py" + (f" --profile {profile}" if profile else "")
              + "\n       or serve a cached profile: set default_profile in config/llm_config.yaml "
              "(python3 scripts/download_models.py --list shows them)", file=sys.stderr)
        return 1
    print(f"cached under {os.path.join(hf_home, 'hub')}: {', '.join(models)}")
    return 0


def main():
    args = parse_args()
    if args.ready_timeout:
        print(ready_timeout(profile=args.profile))
        return
    if args.list:
        list_profiles()
        return

    _load_dotenv()
    if args.check:
        sys.exit(check_cached(args.profile, args.all))
    if snapshot_download is None:
        print("[ERROR] huggingface-hub package is not installed on the host.")
        print("Please install it using: pip install huggingface-hub")
        sys.exit(1)
    token = os.environ.get("HF_TOKEN")
    # HF_CACHE (mounted at /models/huggingface) is the HF home; the hub root the
    # container reads (model_cache_dir /models/huggingface/hub) is its hub/ child --
    # snapshot_download's cache_dir IS the hub root, so download into <HF_CACHE>/hub
    hf_home = os.path.expandvars(os.environ.get("HF_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    cache_dir = os.path.join(hf_home, "hub")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Hugging Face Cache Dir: {cache_dir}")
    if token:
        print("HF_TOKEN detected and active.")
    else:
        print("WARN: HF_TOKEN is not set in environment. Gated repositories might fail to download.")

    models = load_models(profile=args.profile, only_default=not args.all)
    print(f"{len(models)} model(s) selected from config/llm_config.yaml")

    for model in models:
        print("\n" + "=" * 80)
        print(f"Downloading/Verifying: {model}")
        print("=" * 80)
        try:
            snapshot_download(
                repo_id=model,
                cache_dir=cache_dir,
                token=token,
                resume_download=True
            )
            print(f"[SUCCESS] {model} is ready and fully cached.")
        except Exception as e:
            print(f"[ERROR] Failed to download {model}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
