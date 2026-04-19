"""
zkp/worker.py
─────────────
Sterile subprocess worker for EZKL operations.

WHY RAYON_NUM_THREADS IS NOT LIMITED HERE
──────────────────────────────────────────────────────────────────────────
The original reason for RAYON_NUM_THREADS=1 was to prevent Rayon from
racing with Uvicorn/PyTorch threads in the *parent* process. That concern
does not apply here: worker.py is a completely separate OS process with its
own clean Rust runtime. No thread handles are inherited from Uvicorn.

Limiting Rayon to 1 thread in a subprocess gains nothing and costs
everything — Halo2's MSM and FFT operations are designed to run in
parallel. At 1 thread the prover runs ~50-100x slower, easily exceeding
any reasonable timeout.

We DO keep OMP_NUM_THREADS=1 because libomp (OpenMP) and Rayon can
compete for the same CPU core affinity slots on Windows. Silencing OpenMP
lets Rayon claim all cores cleanly.

THREE RULES OF STERILITY (must never be broken)
──────────────────────────────────────────────────────────────────────────
1. Set thread-limit env vars BEFORE any import. C/Rust runtimes read them
   from the OS env at DLL load time — setting after import has no effect.
2. Never import PyTorch, scikit-learn, or anything that loads libomp.dll.
   One OpenMP import and Rayon's thread pool collides.
3. Never import config.py — it may pull in PyTorch transitively.
──────────────────────────────────────────────────────────────────────────
"""

# ── Rule 1: thread limits BEFORE any other import ────────────────────────────
import os
# Do NOT set RAYON_NUM_THREADS here — let Rayon use all available cores.
# Halo2's MSM/FFT proof generation requires parallel threads to complete
# in reasonable time. Limiting to 1 thread causes 600s+ timeouts.
os.environ["OMP_NUM_THREADS"] = "1"   # silence OpenMP so it doesn't race Rayon
os.environ["MKL_NUM_THREADS"] = "1"   # silence Intel MKL for same reason

# ── Rule 2: only safe stdlib + ezkl ──────────────────────────────────────────
import sys
import json
import ezkl   # pure Rust extension, no OpenMP dependency


def run():
    if len(sys.argv) < 2:
        print("Usage: worker.py <manifest.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        job = json.load(f)

    verb = job["verb"]
    args = job["args"]   # all paths are absolute (set by caller)

    print(f"[worker] START  verb={verb}", flush=True)

    if verb == "witness":
        # gen_witness(data_path, compiled_circuit, output_path)
        ezkl.gen_witness(args[0], args[1], args[2])

    elif verb == "prove":
        # prove(witness, compiled_circuit, pk_path, proof_path, srs_path)
        if not os.path.exists(args[4]):
            raise FileNotFoundError(f"SRS file missing: {args[4]}")
        ezkl.prove(args[0], args[1], args[2], args[3], args[4])

    elif verb == "verify":
        # verify(proof_path, settings_path, vk_path, srs_path)
        ok = ezkl.verify(args[0], args[1], args[2], args[3])
        sys.exit(0 if ok else 2)   # exit 2 = proof invalid, not a crash

    else:
        raise ValueError(f"Unknown verb: {verb}")

    print(f"[worker] SUCCESS verb={verb}", flush=True)


if __name__ == "__main__":
    try:
        run()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[worker] ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)