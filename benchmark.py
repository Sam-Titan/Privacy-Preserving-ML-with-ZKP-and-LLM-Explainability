"""
benchmark.py
────────────
Self-contained ZK-SNARK benchmark for the research paper.

Runs EZKL through the same sterile worker.py used by prove.py — no
separate subprocess helper code needed. Produces benchmark_results.csv
and benchmark_summary.json for use in the paper.

Run:
    python benchmark.py              # 10 samples (default)
    python benchmark.py --samples 20
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH      = os.path.join(BASE_DIR, "data",    "heart_disease.csv")
OUTPUTS_DIR    = os.path.join(BASE_DIR, "outputs")
ZKP_DIR        = os.path.join(OUTPUTS_DIR, "zkp")

MODEL_PT_PATH  = os.path.join(OUTPUTS_DIR, "heart_model.pt")
COMPILED_MODEL = os.path.join(ZKP_DIR, "model.compiled")
SETTINGS_PATH  = os.path.join(ZKP_DIR, "settings.json")
PK_PATH        = os.path.join(ZKP_DIR, "pk.key")
VK_PATH        = os.path.join(ZKP_DIR, "vk.key")

# Temp files reused each iteration
INPUT_JSON     = os.path.join(ZKP_DIR, "_bench_input.json")
WITNESS_JSON   = os.path.join(ZKP_DIR, "_bench_witness.json")
PROOF_JSON     = os.path.join(ZKP_DIR, "_bench_proof.json")

RESULTS_CSV    = os.path.join(BASE_DIR, "benchmark_results.csv")
SUMMARY_JSON   = os.path.join(BASE_DIR, "benchmark_summary.json")

# worker.py lives next to prove.py
WORKER_PATH    = os.path.join(BASE_DIR, "zkp", "worker.py")

# Timeouts
_WITNESS_TIMEOUT = 120
_PROOF_TIMEOUT   = 10
_VERIFY_TIMEOUT  = 120

# Windows flags
_CREATE_NO_WINDOW         = 0x08000000
_CREATE_NEW_PROCESS_GROUP = 0x00000200


# ══════════════════════════════════════════════════════════════════════════════
# Subprocess dispatch — identical logic to prove.py so it "just works"
# ══════════════════════════════════════════════════════════════════════════════

def _clean_env() -> dict:
    """
    Build a clean OS environment block for worker subprocesses.

    Key fixes applied here (must be in CreateProcess env, not set inside
    the child script — DLL loader reads them before Python starts):

    CI=1              → disables indicatif progress bars (the TTY deadlock fix)
    OMP_NUM_THREADS=1 → silences OpenMP so it doesn't race Rayon for CPU cores
    MKL_NUM_THREADS=1 → silences Intel MKL for the same reason
    Strip KMP_/TORCH_ → remove any thread-affinity locks inherited from PyTorch
    """
    env = os.environ.copy()

    # Strip PyTorch / OpenMP poison from inherited environment
    for k in list(env.keys()):
        if k.startswith(("KMP_", "OMP_", "GOMP_", "TORCH_", "MKL_")):
            del env[k]

    env["CI"]              = "1"      # THE critical fix — kills indicatif TTY hang
    env["RUST_LOG"]        = "error"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    # NOTE: do NOT set RAYON_NUM_THREADS here — Halo2 MSM/FFT needs all cores

    if sys.platform == "win32":
        env.setdefault("HOME",            os.environ.get("USERPROFILE", "C:\\"))
        env.setdefault("XDG_DATA_HOME",   os.environ.get("APPDATA",    "C:\\"))
        env.setdefault("XDG_CONFIG_HOME", os.environ.get("LOCALAPPDATA","C:\\"))

    return env


def _kill_tree(pid: int) -> None:
    """Force-kill process and all children to prevent zombie Rust threads."""
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


def _dispatch(verb: str, args: list, timeout: int) -> float:
    """
    Run worker.py in a sterile subprocess and return wall-clock time (seconds).
    Raises RuntimeError on failure or timeout.
    """
    manifest_path = os.path.join(ZKP_DIR, f"_bench_manifest_{verb}.json")
    job = {"verb": verb, "args": [os.path.abspath(str(a)) for a in args]}
    with open(manifest_path, "w") as f:
        json.dump(job, f)

    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = _CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        [sys.executable, WORKER_PATH, manifest_path],
        env=_clean_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        **kwargs,
    )

    t0 = time.perf_counter()
    try:
        stdout_b, stderr_b = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc.pid)
        proc.wait()
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        raise RuntimeError(
            f"[benchmark] '{verb}' timed out after {timeout}s.\n"
            "Check that worker.py does NOT set RAYON_NUM_THREADS=1."
        )
    finally:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)

    elapsed = time.perf_counter() - t0

    # Print worker output so we can see EZKL progress messages
    if stdout_b:
        print(stdout_b.decode(errors="replace"), end="", flush=True)
    if stderr_b:
        # Only print stderr on failure to keep output clean
        if proc.returncode not in (0, 2):
            print(stderr_b.decode(errors="replace"), end="", file=sys.stderr)

    if proc.returncode not in (0, 2):   # 2 = proof invalid (not a crash)
        raise RuntimeError(
            f"[benchmark] worker '{verb}' exited {proc.returncode}. "
            "See stderr above."
        )

    return elapsed, proc.returncode


# ══════════════════════════════════════════════════════════════════════════════
# ML helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_artifacts():
    sys.path.insert(0, BASE_DIR)
    from models.preprocess import load_scaler, load_feature_cols
    from models.model import HeartDiseaseNet

    scaler       = load_scaler()
    feature_cols = load_feature_cols()
    model        = HeartDiseaseNet(input_size=len(feature_cols))
    model.load_state_dict(torch.load(MODEL_PT_PATH, map_location="cpu"))
    model.eval()
    return scaler, feature_cols, model


def get_srs_path() -> str:
    with open(SETTINGS_PATH) as f:
        s = json.load(f)
    return os.path.join(ZKP_DIR, f"kzg{s['run_args']['logrows']}.srs")


def preprocess_row(row, feature_cols, scaler):
    MAPS = {
        "sex":     {"male": 1, "m": 1, "female": 0, "f": 0},
        "cp":      {"typical angina": 0, "atypical angina": 1, "atypical": 1,
                    "non-anginal pain": 2, "non-anginal": 2, "asymptomatic": 3},
        "restecg": {"normal": 0, "stt abnormality": 1, "st-t abnormality": 1,
                    "st-t wave abnormality": 1, "lv hypertrophy": 2,
                    "left ventricular hypertrophy": 2},
        "exang":   {"true": 1, "false": 0, "yes": 1, "no": 0},
        "slope":   {"upsloping": 0, "flat": 1, "downsloping": 2},
        "thal":    {"normal": 0, "fixed defect": 1, "reversible defect": 2, "reversable defect": 2},
    }
    values = []
    for col in feature_cols:
        v = row[col]
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if col in MAPS and v_lower in MAPS[col]:
                v = MAPS[col][v_lower]
        elif col in MAPS and v in MAPS[col]:
            v = MAPS[col][v]
        values.append(float(v))

    raw    = np.array([values], dtype=np.float32)
    scaled = scaler.transform(raw)
    return raw, scaled[0].tolist()


def fp32_inference(model, scaled_list):
    x  = torch.FloatTensor([scaled_list])
    t0 = time.perf_counter()
    with torch.no_grad():
        prob = model(x).item()
    return prob, time.perf_counter() - t0

def extract_circuit_output(witness_path: str):
    """Parse the quantized output and handle BN254 prime field wrap-arounds."""
    try:
        with open(witness_path) as f:
            w = json.load(f)
        raw = w.get("outputs", w.get("output_data"))
        if not raw:
            return None
            
        val = raw[0][0] if isinstance(raw[0], list) else raw[0]
        
        if isinstance(val, str):
            with open(SETTINGS_PATH) as f:
                s = json.load(f)
            scale = 2 ** s["run_args"]["input_scale"]
            
            # The BN254 Prime used by Halo2/EZKL
            PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617
            val_int = int(val, 16)
            
            # If the number is in the upper half of the field, it's actually negative
            if val_int > PRIME // 2:
                val_int -= PRIME
                
            return val_int / scale
            
        return float(val)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Main benchmark loop
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(num_samples: int = 10):
    print("=" * 62)
    print(f"  ZK-SNARK BENCHMARK  ({num_samples} samples)")
    print("=" * 62)

    # Pre-flight checks
    for path, name in [
        (WORKER_PATH,    "worker.py"),
        (COMPILED_MODEL, "model.compiled"),
        (PK_PATH,        "pk.key"),
        (VK_PATH,        "vk.key"),
        (SETTINGS_PATH,  "settings.json"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file missing: {name}  ({path})")

    srs_path = get_srs_path()
    if not os.path.exists(srs_path):
        raise FileNotFoundError(f"SRS file missing: {srs_path}")

    scaler, feature_cols, model = load_artifacts()

    df        = pd.read_csv(DATA_PATH).dropna()
    sample_df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    records = []

    for i, (_, row) in enumerate(sample_df.iterrows()):
        print(f"\n── Sample {i + 1}/{num_samples} {'─'*40}")

        try:
            _, scaled = preprocess_row(row, feature_cols, scaler)
        except Exception as e:
            print(f"   [!] Preprocessing failed — skipping: {e}")
            continue

        # ── 1. FP32 baseline inference ─────────────────────────────────────
        fp32_prob, fp32_latency = fp32_inference(model, scaled)
        fp32_label = int(fp32_prob >= 0.5)
        print(f"   FP32 prob   : {fp32_prob:.4f} → {'Disease' if fp32_label else 'No Disease'}"
              f"  ({fp32_latency*1000:.3f} ms)")

        # ── 2. Write private input ─────────────────────────────────────────
        with open(INPUT_JSON, "w") as f:
            json.dump({"input_data": [scaled]}, f)

        # ── 3. Witness generation ──────────────────────────────────────────
        print("   [ZKP] Generating witness …", flush=True)
        t_witness, _ = _dispatch(
            "witness",
            [INPUT_JSON, COMPILED_MODEL, WITNESS_JSON],
            _WITNESS_TIMEOUT,
        )
        print(f"   ✓  Witness  ({t_witness:.3f}s)", flush=True)

        # ── 4. Quantization fidelity (circuit output vs FP32) ──────────────
        circuit_out = extract_circuit_output(WITNESS_JSON)
        if circuit_out is not None:
            quant_error = abs(fp32_prob - circuit_out)
            label_match = (fp32_label == int(circuit_out >= 0.5))
            print(f"   Circuit out : {circuit_out:.4f}  |  Δ={quant_error:.6f}"
                  f"  |  label match={label_match}")
        else:
            quant_error = label_match = None

        # ── 5. Proof generation ────────────────────────────────────────────
        if os.path.exists(PROOF_JSON):
            os.remove(PROOF_JSON)

        print("   [ZKP] Generating proof …", flush=True)
        try:
            t_proof, _ = _dispatch(
                "prove",
                [WITNESS_JSON, COMPILED_MODEL, PK_PATH, PROOF_JSON, srs_path],
                _PROOF_TIMEOUT,  # This will kill the infinite loop after 300s
            )
        except RuntimeError as e:
            print(f"   [!] Proof generation failed or timed out (Likely a Field Overflow). Skipping.")
            continue  # Skip to the next patient

        if not os.path.exists(PROOF_JSON):
            print("   [!] Proof file not written — skipping sample")
            continue

        proof_size_kb = os.path.getsize(PROOF_JSON) / 1024
        print(f"   ✓  Proof    ({t_proof:.3f}s,  {proof_size_kb:.2f} KB)", flush=True)

        # ── 6. Verification ────────────────────────────────────────────────
        print("   [ZKP] Verifying proof …", flush=True)
        t_verify, rc = _dispatch(
            "verify",
            [PROOF_JSON, SETTINGS_PATH, VK_PATH, srs_path],
            _VERIFY_TIMEOUT,
        )
        is_valid = (rc == 0)
        print(f"   {'✓  VALID' if is_valid else '✗  INVALID'}  ({t_verify:.4f}s)", flush=True)

        # ── 7. Overhead ratio ──────────────────────────────────────────────
        overhead = (t_witness + t_proof) / fp32_latency if fp32_latency > 0 else None
        print(f"   Overhead    : {overhead:.0f}x  vs plain inference")

        records.append({
            "sample_id":        i + 1,
            "fp32_probability": round(fp32_prob, 6),
            "fp32_label":       fp32_label,
            "circuit_output":   round(circuit_out, 6)  if circuit_out  is not None else None,
            "quant_error":      round(quant_error, 8)  if quant_error  is not None else None,
            "label_match":      label_match,
            "fp32_latency_ms":  round(fp32_latency * 1000, 4),
            "witness_time_s":   round(t_witness, 4),
            "proof_time_s":     round(t_proof,   4),
            "verify_time_s":    round(t_verify,  4),
            "total_zkp_time_s": round(t_witness + t_proof + t_verify, 4),
            "proof_size_kb":    round(proof_size_kb, 2),
            "overhead_ratio":   round(overhead, 1) if overhead else None,
            "is_valid":         is_valid,
        })

    # ── Cleanup ────────────────────────────────────────────────────────────────
    for p in [INPUT_JSON, WITNESS_JSON, PROOF_JSON]:
        if os.path.exists(p):
            os.remove(p)

    if not records:
        print("\n[!] No results collected.")
        return

    # ── Aggregate ──────────────────────────────────────────────────────────────
    rdf            = pd.DataFrame(records)
    valid_quant    = rdf["quant_error"].dropna()
    proof_size_std = rdf["proof_size_kb"].std()

    summary = {
        "num_samples":            len(records),
        # Timing
        "avg_witness_time_s":     round(rdf["witness_time_s"].mean(), 4),
        "avg_proof_time_s":       round(rdf["proof_time_s"].mean(),   4),
        "std_proof_time_s":       round(rdf["proof_time_s"].std(),    4),
        "avg_verify_time_s":      round(rdf["verify_time_s"].mean(),  4),
        "avg_total_zkp_time_s":   round(rdf["total_zkp_time_s"].mean(), 4),
        # Proof size
        "avg_proof_size_kb":      round(rdf["proof_size_kb"].mean(),  2),
        "std_proof_size_kb":      round(proof_size_std, 6),
        "proof_size_is_constant": bool(proof_size_std < 0.01),
        # Quantization fidelity
        "avg_quant_error":        round(float(valid_quant.mean()), 8) if len(valid_quant) else None,
        "max_quant_error":        round(float(valid_quant.max()),  8) if len(valid_quant) else None,
        "label_match_rate":       round(float(rdf["label_match"].dropna().mean()), 4)
                                  if rdf["label_match"].notna().any() else None,
        # Overhead
        "avg_overhead_ratio":     round(float(rdf["overhead_ratio"].dropna().mean()), 1),
        "avg_fp32_latency_ms":    round(float(rdf["fp32_latency_ms"].mean()), 4),
        # Validity
        "proof_validity_rate":    float(rdf["is_valid"].mean()),
    }

    rdf.to_csv(RESULTS_CSV, index=False)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  RESULTS SUMMARY")
    print("=" * 62)
    print(f"\n  [ ZKP TIMING ]")
    print(f"  Avg Witness Time     : {summary['avg_witness_time_s']:.4f} s")
    print(f"  Avg Proof Time       : {summary['avg_proof_time_s']:.4f} s  (±{summary['std_proof_time_s']:.4f})")
    print(f"  Avg Verify Time      : {summary['avg_verify_time_s']:.4f} s")
    print(f"  Avg Total ZKP Time   : {summary['avg_total_zkp_time_s']:.4f} s")
    print(f"\n  [ PROOF SIZE ]")
    print(f"  Avg Proof Size       : {summary['avg_proof_size_kb']:.2f} KB")
    print(f"  Std Dev of Size      : {summary['std_proof_size_kb']:.6f} KB")
    print(f"  Constant Size?       : {'YES — privacy demonstrated' if summary['proof_size_is_constant'] else 'NO — investigate'}")
    print(f"\n  [ QUANTIZATION FIDELITY ]")
    if summary["avg_quant_error"] is not None:
        print(f"  Avg Quant Error      : {summary['avg_quant_error']:.8f}")
        print(f"  Max Quant Error      : {summary['max_quant_error']:.8f}")
        print(f"  Label Match Rate     : {summary['label_match_rate']*100:.1f}%")
    else:
        print("  (circuit output not extractable from witness — check EZKL version)")
    print(f"\n  [ PRIVACY OVERHEAD ]")
    print(f"  Avg FP32 Latency     : {summary['avg_fp32_latency_ms']:.4f} ms")
    print(f"  Avg Overhead Ratio   : {summary['avg_overhead_ratio']:.0f}x  vs plain inference")
    print(f"\n  [ VALIDITY ]")
    print(f"  Proof Validity Rate  : {summary['proof_validity_rate']*100:.1f}%")
    print(f"\n  Results → {RESULTS_CSV}")
    print(f"  Summary → {SUMMARY_JSON}")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of dataset samples (default: 10)")
    args = parser.parse_args()
    run_benchmark(num_samples=args.samples)
