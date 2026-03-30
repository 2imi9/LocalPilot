"""
Run both experiments back-to-back: enhanced v3 first, then baseline.
Automatically resets train.py between runs.

Usage:
    .venv/Scripts/python.exe run_both.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
CLEAN_COMMIT = "9682daa"  # clean karpathy baseline commit


def reset_train():
    """Reset train.py to clean karpathy baseline."""
    subprocess.run(["git", "checkout", CLEAN_COMMIT, "--", "train.py"],
                   cwd=str(ROOT), capture_output=True)
    print(f"[run_both] train.py reset to {CLEAN_COMMIT}")


def kill_servers():
    """Kill any lingering llama-server or python training processes."""
    subprocess.run(
        ["powershell", "-Command",
         "Get-Process llama* -ErrorAction SilentlyContinue | Stop-Process -Force"],
        capture_output=True)
    time.sleep(3)


def run_experiment(script, label):
    """Run an experiment script to completion."""
    print(f"\n{'='*60}")
    print(f"  [run_both] Starting {label}")
    print(f"  [run_both] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [PYTHON, script],
        cwd=str(ROOT),
        timeout=86400,  # 24h safety cap
    )
    elapsed = time.time() - start
    hours = elapsed / 3600

    print(f"\n{'='*60}")
    print(f"  [run_both] {label} finished")
    print(f"  [run_both] Exit code: {result.returncode}")
    print(f"  [run_both] Wall time: {hours:.1f}h ({elapsed:.0f}s)")
    print(f"  [run_both] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    return result.returncode


def main():
    print("="*60)
    print("  Autoresearch -- Running both conditions back-to-back")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # --- Phase 1: Enhanced v3 ---
    reset_train()
    kill_servers()
    rc1 = run_experiment("experiments/run_enhanced_v3.py", "Enhanced v3 (LLM-guided)")

    # --- Reset between runs ---
    kill_servers()
    reset_train()
    time.sleep(5)

    # --- Phase 2: Baseline ---
    rc2 = run_experiment("experiments/run_baseline_v2.py", "Baseline (random search)")

    # --- Summary ---
    kill_servers()
    print("\n" + "="*60)
    print("  BOTH RUNS COMPLETE")
    print(f"  Enhanced v3: exit code {rc1}")
    print(f"  Baseline:    exit code {rc2}")
    print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Quick results summary
    for label, tsv in [("Enhanced v3", "results_enhanced_v3.tsv"),
                       ("Baseline", "results_baseline_v2.tsv")]:
        p = ROOT / tsv
        if p.exists():
            lines = p.read_text().strip().splitlines()
            n = len(lines) - 1  # minus header
            keeps = sum(1 for l in lines[1:] if "\tkeep\t" in l)
            # Find best
            best = 99.0
            for l in lines[1:]:
                parts = l.split("\t")
                if len(parts) > 3 and parts[3] == "keep":
                    val = float(parts[1])
                    if val < best:
                        best = val
            print(f"\n  {label}: {n} experiments, {keeps} keeps, best={best:.6f}")

    print("\nDone! Results in results_enhanced_v3.tsv and results_baseline_v2.tsv")


if __name__ == "__main__":
    main()
