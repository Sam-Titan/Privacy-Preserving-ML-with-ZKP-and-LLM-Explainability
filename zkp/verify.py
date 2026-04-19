"""
zkp/verify.py
─────────────
Verify the SNARK proof via worker.py subprocess.
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROOF_PATH, VK_PATH, SETTINGS_PATH, ZKP_DIR

_WORKER_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
_VERIFY_TIMEOUT = 120
_CREATE_NO_WINDOW = 0x08000000
_CREATE_NEW_PROCESS_GROUP = 0x00000200  # <-- ADD THIS


def _clean_env() -> dict:
    env = os.environ.copy()

    # ── 1. STRIP PYTORCH/OPENMP POISON ──
    # Remove inherited thread-affinity locks that cause Rayon to deadlock.
    for k in list(env.keys()):
        if k.startswith(("KMP_", "OMP_", "GOMP_", "TORCH_", "MKL_")):
            del env[k]

    # ── 2. FIX THE RUST TTY DEADLOCK ──
    # Forces `indicatif` to disable interactive progress bars. Without this,
    # ezkl hangs infinitely trying to query a non-existent Windows console.
    env["CI"] = "1"
    env["RUST_LOG"] = "error"

    # Re-apply safe limits just in case a secondary C-library loads
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    if sys.platform == "win32":
        env.setdefault("HOME",            os.environ.get("USERPROFILE", "C:\\"))
        env.setdefault("XDG_DATA_HOME",   os.environ.get("APPDATA",    "C:\\"))
        env.setdefault("XDG_CONFIG_HOME", os.environ.get("LOCALAPPDATA","C:\\"))

    return env


def _get_srs_path() -> str:
    with open(SETTINGS_PATH) as f:
        s = json.load(f)
    logrows  = s["run_args"]["logrows"]
    srs_file = f"kzg{logrows}.srs"
    if sys.platform == "win32":
        return os.path.join(ZKP_DIR, srs_file)
    elif sys.platform == "darwin":
        return os.path.expanduser(
            f"~/Library/Application Support/ezkl/srs/{srs_file}"
        )
    else:
        return os.path.expanduser(f"~/.local/share/ezkl/srs/{srs_file}")


def _kill_tree(pid: int) -> None:
    """Force-kill the entire process tree to prevent zombie Rust threads."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
        )
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def verify_proof(
    proof_path:    str = PROOF_PATH,
    vk_path:       str = VK_PATH,
    settings_path: str = SETTINGS_PATH,
) -> dict:
    if not os.path.exists(proof_path):
        return {
            "is_valid":      False,
            "verify_time_s": 0.0,
            "message":       f"Proof file not found: {proof_path}",
        }

    print("[ZKP] Verifying proof …", flush=True)
    t0 = time.perf_counter()

    srs_path      = _get_srs_path()
    manifest_path = os.path.join(ZKP_DIR, "_manifest_verify.json")

    job = {
        "verb": "verify",
        "args": [
            os.path.abspath(proof_path),
            os.path.abspath(settings_path),
            os.path.abspath(vk_path),
            os.path.abspath(srs_path),
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(job, f)

    kwargs = {}
    if sys.platform == "win32":
        # <-- UPDATE THIS LINE TO BITWISE OR BOTH FLAGS
        kwargs["creationflags"] = _CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        [sys.executable, _WORKER_PATH, manifest_path],
        env=_clean_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        **kwargs,
    )

    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=_VERIFY_TIMEOUT)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"   ✗  Verify timed out after {elapsed:.1f}s — killing process tree …")
        _kill_tree(proc.pid)
        proc.wait()
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        return {
            "is_valid":      False,
            "verify_time_s": round(elapsed, 3),
            "message":       f"Verification timed out after {_VERIFY_TIMEOUT}s.",
        }
    finally:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)

    elapsed = time.perf_counter() - t0

    # exit 0 = valid, exit 2 = invalid proof (not a crash), exit 1 = error
    is_valid = proc.returncode == 0
    is_crash = proc.returncode == 1

    print(f"   {'✓  VALID' if is_valid else '✗  INVALID'}  ({elapsed:.3f}s)", flush=True)

    if is_crash and stderr_bytes:
        print(f"   [ZKP] Worker error: {stderr_bytes.decode(errors='replace')[-300:]}")

    return {
        "is_valid":      is_valid,
        "verify_time_s": round(elapsed, 3),
        "message": (
            "Proof is cryptographically valid. "
            "The prediction was computed correctly on private input data."
            if is_valid else
            "Proof verification failed."
            + (f" Detail: {stderr_bytes.decode(errors='replace')[-150:]}"
               if stderr_bytes else "")
        ),
    }


if __name__ == "__main__":
    print(json.dumps(verify_proof(), indent=2))


# """
# zkp/verify.py
# ─────────────
# SIMULATION MODE FOR DEMO
# Mocks the verification of the dummy SNARK proof.
# """

# import json
# import os
# import time
# import sys

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config import PROOF_PATH, SETTINGS_PATH, VK_PATH

# def verify_proof(
#     proof_path:    str = PROOF_PATH,
#     vk_path:       str = VK_PATH,
#     settings_path: str = SETTINGS_PATH,
# ) -> dict:
    
#     print("[ZKP] Verifying proof ...", flush=True)
#     t0 = time.perf_counter()
    
#     if not os.path.exists(proof_path):
#         return {
#             "is_valid":      False,
#             "verify_time_s": 0.0,
#             "message":       f"Proof file not found: {proof_path}",
#         }

#     # Simulate cryptographic verification delay
#     time.sleep(0.4)
#     elapsed = time.perf_counter() - t0

#     print(f"   ✓  VALID  ({elapsed:.3f}s)", flush=True)

#     return {
#         "is_valid":      True,
#         "verify_time_s": round(elapsed, 3),
#         "message":       "Proof is cryptographically valid. The prediction was computed correctly on private input data."
#     }

# if __name__ == "__main__":
#     print(json.dumps(verify_proof(), indent=2))