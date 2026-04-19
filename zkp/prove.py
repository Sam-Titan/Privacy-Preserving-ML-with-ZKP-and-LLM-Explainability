"""
zkp/prove.py
────────────
Orchestrates witness + proof generation by delegating to worker.py.

WHY RAYON_NUM_THREADS IS NOT CAPPED
─────────────────────────────────────────────────────────────────────────
The env cap of RAYON_NUM_THREADS=1 was originally added to prevent a
race condition between Uvicorn/PyTorch and Rayon in the *parent* process.
worker.py is a completely separate OS process — it inherits no thread
handles from Uvicorn. Capping it to 1 thread forces Halo2's MSM/FFT
operations to run sequentially, which takes 600s+ instead of ~5s.

We keep OMP_NUM_THREADS=1 to silence OpenMP (libomp.dll) so it does not
compete with Rayon for CPU core affinity slots on Windows.

WHY stdout=PIPE (NOT a file handle)
─────────────────────────────────────────────────────────────────────────
Passing stdout=open(logfile) with CREATE_NO_WINDOW silently breaks Windows
handle inheritance — the child's first write blocks forever, causing an
unrecoverable deadlock. PIPE is always safe; Python's communicate() drains
it internally.

HOW ZOMBIE CLEANUP WORKS
─────────────────────────────────────────────────────────────────────────
On timeout, Python's proc.kill() only signals the top-level process. On
Windows, Rust threads often ignore this signal and keep running. We use
`taskkill /F /T /PID` to force-kill the entire process TREE, ensuring no
orphan Rust threads persist in Task Manager.
─────────────────────────────────────────────────────────────────────────
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    COMPILED_CIRCUIT, PK_PATH, SETTINGS_PATH,
    WITNESS_PATH, PROOF_PATH, INPUT_JSON_PATH, ZKP_DIR,
)

_WORKER_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
_WITNESS_TIMEOUT = 120    # seconds
_PROOF_TIMEOUT   = 300    # seconds — parallel Rayon finishes in ~5-30s on modern hardware

# Windows process creation flags
_CREATE_NO_WINDOW         = 0x08000000
_CREATE_NEW_PROCESS_GROUP = 0x00000200   # Detaches OS thread scheduling
_ABOVE_NORMAL_PRIORITY    = 0x00008000
_HIGH_PRIORITY            = 0x00000080

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
    """
    Force-kill a process and ALL its children on Windows.

    proc.kill() only signals the top-level Python process. On Windows, Rust
    threads launched by the Rust runtime (Rayon workers, Tokio threads) often
    ignore this signal and persist as zombies at 0% CPU in Task Manager.
    `taskkill /F /T` kills the entire job tree — no orphans survive.
    """
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,   # don't pollute our terminal
        )
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _dispatch(verb: str, args: list, step_label: str, timeout: int,
              high_priority: bool = False) -> None:
    """
    Write a JSON manifest and run worker.py in a sterile subprocess.

    Uses Popen + communicate() so we can force-kill the full process tree
    on timeout, preventing zombie Rust threads from lingering in Task Manager.
    """
    manifest_path = os.path.join(ZKP_DIR, f"_manifest_{verb}.json")
    job = {"verb": verb, "args": [os.path.abspath(str(a)) for a in args]}
    with open(manifest_path, "w") as f:
        json.dump(job, f)

    # Build creation flags
    flags = _CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP
    if sys.platform == "win32" and high_priority:
        flags |= _HIGH_PRIORITY   # boost scheduling for CPU-intensive proof gen

    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = flags

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
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(
            f"\n[ZKP] {step_label} timed out after {timeout}s — "
            f"killing process tree (PID {proc.pid}) …",
            flush=True,
        )
        _kill_tree(proc.pid)
        proc.wait()   # reap the now-dead process
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        raise RuntimeError(
            f"{step_label} timed out after {timeout}s.\n"
            "If this happens consistently, check that worker.py does NOT set\n"
            "RAYON_NUM_THREADS=1 (that caps Rayon to a single thread and\n"
            "makes Halo2 prove take 600s+ instead of ~5-30s)."
        )
    finally:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)

    # Stream output to parent terminal after child exits cleanly
    if stdout_bytes:
        print(stdout_bytes.decode(errors="replace"), end="", flush=True)
    if stderr_bytes:
        print(stderr_bytes.decode(errors="replace"), end="", file=sys.stderr, flush=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"{step_label} worker exited with code {proc.returncode}. "
            "See stderr output above."
        )


def generate_proof(scaled_input: list) -> dict:
    """
    Generate a ZK witness + SNARK proof for the given scaled feature vector.

    Parameters
    ----------
    scaled_input : list[float]
        StandardScaler-transformed features (length = number of model inputs).
    """
    os.makedirs(ZKP_DIR, exist_ok=True)

    with open(INPUT_JSON_PATH, "w") as f:
        json.dump({"input_data": [scaled_input]}, f)

    if os.path.exists(PROOF_PATH):
        os.remove(PROOF_PATH)

    srs_path = _get_srs_path()
    if not os.path.exists(srs_path):
        raise FileNotFoundError(
            f"SRS file not found: {srs_path}\n"
            "Run  python -m zkp.setup  to download it."
        )

    t_start = time.perf_counter()

    # ── Witness (fast, no priority boost needed) ──────────────────────────────
    print("[ZKP] Generating witness …", flush=True)
    t0 = time.perf_counter()
    _dispatch(
        verb="witness",
        args=[INPUT_JSON_PATH, COMPILED_CIRCUIT, WITNESS_PATH],
        step_label="gen-witness",
        timeout=_WITNESS_TIMEOUT,
        high_priority=False,
    )
    witness_time = time.perf_counter() - t0
    print(f"   ✓  Witness ready  ({witness_time:.2f}s)", flush=True)

    # ── Proof (CPU-intensive — high priority + full Rayon parallelism) ────────
    print("[ZKP] Generating SNARK proof …", flush=True)
    t1 = time.perf_counter()
    _dispatch(
        verb="prove",
        args=[WITNESS_PATH, COMPILED_CIRCUIT, PK_PATH, PROOF_PATH, srs_path],
        step_label="prove",
        timeout=_PROOF_TIMEOUT,
        high_priority=True,   # boost scheduling priority for crypto workload
    )
    proof_time = time.perf_counter() - t1

    if not os.path.exists(PROOF_PATH):
        raise RuntimeError(
            f"Proof file missing at {PROOF_PATH} even though worker exited 0. "
            "Check ezkl version compatibility."
        )

    proof_size_kb = os.path.getsize(PROOF_PATH) / 1024
    print(f"   ✓  Proof ready  ({proof_time:.2f}s,  {proof_size_kb:.1f} KB)", flush=True)

    return {
        "proof_path":     PROOF_PATH,
        "proof_size_kb":  round(proof_size_kb, 2),
        "witness_time_s": round(witness_time, 3),
        "proof_time_s":   round(proof_time, 3),
        "total_time_s":   round(time.perf_counter() - t_start, 3),
    }


if __name__ == "__main__":
    from models.preprocess import load_feature_cols
    result = generate_proof([0.0] * len(load_feature_cols()))
    print(json.dumps(result, indent=2))


# """
# zkp/prove.py
# ────────────
# SIMULATION MODE FOR DEMO
# Bypasses the OS-level EZKL deadlock by mocking the cryptographic proof generation.
# """

# import json
# import os
# import time
# import sys

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config import PROOF_PATH, INPUT_JSON_PATH, ZKP_DIR, WITNESS_PATH

# def generate_proof(scaled_input: list) -> dict:
#     os.makedirs(ZKP_DIR, exist_ok=True)

#     # 1. Save the input so the rest of the pipeline can use it
#     with open(INPUT_JSON_PATH, "w") as f:
#         json.dump({"input_data": [scaled_input]}, f)
        
#     # Mock the witness file
#     with open(WITNESS_PATH, "w") as f:
#         json.dump({"mock": "witness_data"}, f)

#     t_start = time.perf_counter()

#     print("[ZKP] Generating witness ...", flush=True)
#     time.sleep(0.3) # Simulate fast witness time
#     witness_time = time.perf_counter() - t_start
#     print(f"   ✓  Witness ready  ({witness_time:.2f}s)", flush=True)

#     print("[ZKP] Generating SNARK proof ...", flush=True)
#     print(f"      [Simulating cryptographic math for demo...]", flush=True)
    
#     # Simulate the heavy computation delay (approx 3-5 seconds)
#     time.sleep(3.8) 
    
#     # 2. Generate a valid-looking dummy proof
#     if os.path.exists(PROOF_PATH):
#         os.remove(PROOF_PATH)
#     with open(PROOF_PATH, "w") as f:
#         json.dump({
#             "proof": "0x" + "0" * 64 + " simulated_zk_snark_proof_for_windows_demo",
#             "instances": []
#         }, f)

#     proof_time = time.perf_counter() - (t_start + witness_time)
#     proof_size_kb = 12.4  # Standard size for a logrows=12 circuit

#     print(f"   ✓  Proof ready  ({proof_time:.2f}s,  {proof_size_kb:.1f} KB)", flush=True)

#     return {
#         "proof_path":     PROOF_PATH,
#         "proof_size_kb":  proof_size_kb,
#         "witness_time_s": round(witness_time, 3),
#         "proof_time_s":   round(proof_time, 3),
#         "total_time_s":   round(time.perf_counter() - t_start, 3),
#     }

# if __name__ == "__main__":
#     from models.preprocess import load_feature_cols
#     print(json.dumps(generate_proof([0.0] * len(load_feature_cols())), indent=2))