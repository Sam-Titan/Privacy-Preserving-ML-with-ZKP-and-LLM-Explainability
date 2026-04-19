"""
zkp/setup.py
────────────
One-time setup for EZKL.

WHY SUBPROCESS FOR STEPS 4 & 5?
─────────────────────────────────────────────────────────────────────────────
EZKL's SRS fetch uses a Rust async runtime. On Windows it frequently panics
via a Tokio thread error. A Rust `spin::Once` that panics permanently
"poisons" its lock — meaning any subsequent ezkl call in the SAME Python
process also panics with "Once panicked", regardless of RAM or circuit size.

Fix: on Windows, skip the in-process SRS call entirely and do both SRS
fetching (step 4) AND key generation (step 5) in fresh subprocesses, giving
each a completely clean, un-poisoned Rust runtime.
─────────────────────────────────────────────────────────────────────────────

Run: python -m zkp.setup
"""

import asyncio
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['WEBSITE_AUTH_SSL_CERT'] = certifi.where()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CALIBRATION_PATH, COMPILED_CIRCUIT, ONNX_MODEL_PATH,
    PK_PATH, SETTINGS_PATH, VK_PATH, ZKP_DIR,
)
from models.preprocess import load_and_preprocess, load_scaler

try:
    import ezkl
except ImportError:
    raise ImportError("ezkl not installed. Run: pip install ezkl")

INPUT_VISIBILITY  = "private"
OUTPUT_VISIBILITY = "public"
PARAM_VISIBILITY  = "fixed"
N_CALIBRATION     = 600
LOGROWS_CAP       = 16


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _get_ezkl_exe() -> str | None:
    """
    Find the absolute path to the ezkl binary inside the current venv.

    On Windows, `pip install ezkl` drops ezkl.exe into <venv>\\Scripts\\ but
    that folder is often absent from PATH when subprocess.run() executes,
    causing WinError 2. We resolve it via sys.prefix instead.

    Returns None if the binary genuinely cannot be found anywhere (the caller
    will then use the Python-subprocess fallback).
    """
    candidates = [
        os.path.join(sys.prefix, "Scripts", "ezkl.exe"),   # Windows venv
        os.path.join(sys.prefix, "Scripts", "ezkl"),        # Windows (no ext)
        os.path.join(sys.prefix, "bin", "ezkl"),            # Linux / macOS venv
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return shutil.which("ezkl")  # last resort — may return None


def _ezkl_python_subprocess(*args: str, step_label: str = "") -> None:
    """
    Invoke an ezkl operation via a fresh Python subprocess.

    Supports the two verbs used in this file: 'get-srs' and 'setup'.

    Surgical Fix: Wraps the logic in an 'async def main' to ensure the 
    event loop is active before EZKL touches the Rust runtime.
    """
    verb = args[0] if args else ""

    if verb == "get-srs":
        settings_idx = list(args).index("--settings-path") + 1
        settings = args[settings_idx]
        script = "\n".join([
            "import asyncio, inspect, ezkl",
            "async def main():",
            f"    result = ezkl.get_srs(r'{settings}')",
            "    if inspect.isawaitable(result):",
            "        await result",
            "    print('EZKL_SUBPROCESS_OK')",
            "asyncio.run(main())"
        ])

    elif verb == "setup":
        a = list(args)
        circuit = a[a.index("--compiled-circuit") + 1]
        vk      = a[a.index("--vk-path")          + 1]
        pk      = a[a.index("--pk-path")           + 1]
        srs     = a[a.index("--srs-path")          + 1] if "--srs-path" in a else None
        srs_arg = f", srs_path=r'{srs}'" if srs else ""
        script = "\n".join([
            "import asyncio, inspect, ezkl, sys",
            "async def main():",
            f"    result = ezkl.setup(model=r'{circuit}', vk_path=r'{vk}', pk_path=r'{pk}'{srs_arg})",
            "    if inspect.isawaitable(result):",
            "        result = await result",
            "    if result is False: sys.exit(1)",
            "    print('EZKL_SUBPROCESS_OK')",
            "asyncio.run(main())"
        ])

    else:
        raise RuntimeError(
            f"_ezkl_python_subprocess: unsupported verb '{verb}'. "
            "Add a mapping for it or locate the ezkl binary."
        )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True
    )

    if result.returncode == 0 and "EZKL_SUBPROCESS_OK" in result.stdout:
        return

    print(f"\n[!] Python subprocess for '{step_label}' failed (exit {result.returncode})")
    if result.stdout:
        print("  STDOUT:", result.stdout[-800:])
    if result.stderr:
        print("  STDERR:", result.stderr[-800:])
    raise RuntimeError(f"ezkl Python subprocess failed for step: {step_label}")


def _ezkl_cli(*args: str, step_label: str = "") -> None:
    """
    Run an ezkl CLI command in a fresh subprocess (fresh Rust runtime).

    Locates the ezkl binary via _get_ezkl_exe() (venv-aware, avoids WinError 2).
    Falls back to _ezkl_python_subprocess() if the binary cannot be found —
    which is the common case when ezkl is installed as a Python wheel with no
    standalone binary on PATH.
    """
    exe = _get_ezkl_exe()

    if exe:
        cmd = [exe] + list(args)
        print(f"   [i] CLI cmd: {exe} {' '.join(args)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return
        print(f"\n[!] {step_label} failed (exit {result.returncode})")
        if result.stdout:
            print("  STDOUT:", result.stdout[-800:])
        if result.stderr:
            print("  STDERR:", result.stderr[-800:])
        raise RuntimeError(f"ezkl CLI failed: {' '.join(cmd)}")
    else:
        print(f"   [!] ezkl binary not found; using Python subprocess for: {step_label}")
        _ezkl_python_subprocess(*args, step_label=step_label)


# ── SRS path helper ───────────────────────────────────────────────────────────

def _get_srs_path() -> str:
    """Return the EZKL SRS cache path for the current logrows setting."""
    with open(SETTINGS_PATH) as f:
        s = json.load(f)
    logrows = s["run_args"]["logrows"]
    srs_filename = f"kzg{logrows}.srs"

    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        return os.path.join(base, "ezkl", "srs", srs_filename)
    elif sys.platform == "darwin":
        return os.path.expanduser(
            f"~/Library/Application Support/ezkl/srs/{srs_filename}"
        )
    else:
        return os.path.expanduser(f"~/.local/share/ezkl/srs/{srs_filename}")


# ── Main setup routine ────────────────────────────────────────────────────────

async def setup_zkp():
    os.makedirs(ZKP_DIR, exist_ok=True)

    # ── Step 1: Generate settings ─────────────────────────────────────────────
    print("\n[1/5] Generating EZKL settings …")
    py_run_args                   = ezkl.PyRunArgs()
    py_run_args.input_visibility  = INPUT_VISIBILITY
    py_run_args.output_visibility = OUTPUT_VISIBILITY
    py_run_args.param_visibility  = PARAM_VISIBILITY

    ok = ezkl.gen_settings(ONNX_MODEL_PATH, SETTINGS_PATH,
                           py_run_args=py_run_args)
    if not ok:
        raise RuntimeError("gen_settings failed")
    print(f"   ✓  Settings → {SETTINGS_PATH}")

    # ── Step 2: Calibrate + cap logrows ──────────────────────────────────────
    print("[2/5] Calibrating quantization …")
    X, _, _  = load_and_preprocess()
    scaler   = load_scaler()
    X_scaled = scaler.transform(X[:N_CALIBRATION]).tolist()

    with open(CALIBRATION_PATH, "w") as f:
        json.dump({"input_data": X_scaled}, f)

    result = ezkl.calibrate_settings(
        CALIBRATION_PATH, ONNX_MODEL_PATH, SETTINGS_PATH,
        target="accuracy",
    )
    if result is False:
        raise RuntimeError("calibrate_settings returned False")

    # Cap logrows to prevent RAM panic
    with open(SETTINGS_PATH, "r") as f:
        s = json.load(f)

    current_logrows = s.get("run_args", {}).get("logrows", 99)
    if current_logrows > LOGROWS_CAP:
        print(f"   [!] logrows={current_logrows} exceeds cap → forcing to {LOGROWS_CAP}")
        s["run_args"]["logrows"] = LOGROWS_CAP
        with open(SETTINGS_PATH, "w") as f:
            json.dump(s, f, indent=2)
    print(f"   ✓  Calibration done  (logrows={min(current_logrows, LOGROWS_CAP)})")

    # ── Step 3: Compile circuit ───────────────────────────────────────────────
    print("[3/5] Compiling model to ZK circuit …")
    ok = ezkl.compile_circuit(ONNX_MODEL_PATH, COMPILED_CIRCUIT, SETTINGS_PATH)
    if not ok:
        raise RuntimeError("compile_circuit failed")
    print(f"   ✓  Circuit → {COMPILED_CIRCUIT}")

    # ── Step 4: Fetch SRS ─────────────────────────────────────────────────────
    # On Windows we NEVER call ezkl.get_srs() in-process.
    # The Rust Tokio runtime panics and permanently poisons a spin::Once lock,
    # causing every subsequent ezkl call in this process to raise "Once panicked".
    # A fresh subprocess has a clean Rust runtime and sidesteps this entirely.
    print("[4/5] Fetching Structured Reference String (KZG params) …")
    srs_path = _get_srs_path()

    if sys.platform == "win32":
        print("   [i] Windows → fetching SRS in subprocess to avoid Rust runtime poison …")
        if os.path.exists(srs_path):
            print(f"   ✓  SRS already cached → {srs_path}")
        else:
            try:
                _ezkl_python_subprocess(
                    "get-srs", "--settings-path", SETTINGS_PATH,
                    step_label="get-srs"
                )
                print(f"   ✓  SRS ready → {srs_path}")
            except Exception as e:
                # Subprocess may have panicked but still written the file
                if os.path.exists(srs_path):
                    print("   ✓  SRS file found at cache after subprocess → proceeding")
                else:
                    raise RuntimeError(
                        f"Could not fetch SRS: {e}\n"
                        "TIP: Switch to a mobile hotspot if on campus Wi-Fi.\n"
                        f"Expected SRS at: {srs_path}"
                    ) from e
    else:
        # Non-Windows: try in-process first, fall back to subprocess
        try:
            res = ezkl.get_srs(SETTINGS_PATH)
            if inspect.isawaitable(res):
                await res
            print("   ✓  SRS ready (in-process)")
        except Exception as e:
            print(f"   [!] In-process SRS raised: {e}")
            if os.path.exists(srs_path):
                print(f"   ✓  SRS file found at cache → {srs_path}  (error was a false alarm)")
            else:
                print("   [!] SRS not in cache. Trying subprocess fallback …")
                try:
                    _ezkl_python_subprocess(
                        "get-srs", "--settings-path", SETTINGS_PATH,
                        step_label="get-srs"
                    )
                    print("   ✓  SRS ready (subprocess fallback)")
                except Exception as e2:
                    if os.path.exists(srs_path):
                        print("   ✓  SRS file found after subprocess → proceeding")
                    else:
                        raise RuntimeError(
                            f"Could not fetch SRS via any method: {e2}\n"
                            "TIP: Switch to a mobile hotspot if on campus Wi-Fi.\n"
                            f"Expected SRS at: {srs_path}"
                        ) from e2

    # ── Step 5: Key generation — FRESH SUBPROCESS ─────────────────────────────
    # Must run in a subprocess on all platforms: calling ezkl.setup() in this
    # process risks hitting a poisoned Rust runtime from any earlier panic.
    print("[5/5] Generating proving & verification keys …")
    print("   [i] Running in a fresh subprocess (avoids Rust 'Once panicked')")
    print("   [i] This may take 1–5 minutes — please wait …")

    t0 = time.time()

    cli_args = [
        "setup",
        "--compiled-circuit", COMPILED_CIRCUIT,
        "--vk-path",           VK_PATH,
        "--pk-path",           PK_PATH,
    ]
    if os.path.exists(srs_path):
        cli_args += ["--srs-path", srs_path]

    # _ezkl_cli falls back to Python subprocess automatically if binary missing
    _ezkl_cli(*cli_args, step_label="setup (key gen)")

    elapsed = time.time() - t0

    for path, name in [(VK_PATH, "VK"), (PK_PATH, "PK")]:
        if not os.path.exists(path):
            raise RuntimeError(
                f"{name} key not found at {path} even though ezkl exited 0. "
                "Check ezkl version compatibility."
            )

    print(f"   ✓  Keys generated in {elapsed:.1f}s")
    print(f"   ✓  VK → {VK_PATH}")
    print(f"   ✓  PK → {PK_PATH}")
    print("\n✅  ZKP setup complete!  You can now run predictions.\n")


if __name__ == "__main__":
    try:
        asyncio.run(setup_zkp())
    except Exception as e:
        print(f"\n[!] Setup failed: {e}")
        sys.exit(1)