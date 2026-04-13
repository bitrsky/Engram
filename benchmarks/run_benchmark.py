"""
run_benchmark.py -- CLI entry point for Engram benchmarks.

Usage:
    python -m benchmarks.run_benchmark                    # Run all benchmarks
    python -m benchmarks.run_benchmark --layer retrieval  # Specific layer
    python -m benchmarks.run_benchmark --layer pipeline
    python -m benchmarks.run_benchmark --layer conflicts
    python -m benchmarks.run_benchmark --layer e2e
    python -m benchmarks.run_benchmark --output results/run.jsonl
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BENCHMARKS_DIR = Path(__file__).parent

LAYER_MAP = {
    "retrieval": "test_retrieval.py",
    "pipeline": "test_pipeline.py",
    "conflicts": "test_conflicts.py",
    "e2e": "test_e2e.py",
}

# External academic benchmarks (standalone scripts, not pytest)
EXTERNAL_BENCHMARKS = {
    "longmemeval": "bench_longmemeval.py",
    "locomo": "bench_locomo.py",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run Engram benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.run_benchmark                    # All internal benchmarks
  python -m benchmarks.run_benchmark --layer retrieval  # Retrieval only
  python -m benchmarks.run_benchmark --layer pipeline   # Pipeline quality only
  python -m benchmarks.run_benchmark --layer longmemeval  # LongMemEval (external)
  python -m benchmarks.run_benchmark --layer locomo       # LoCoMo (external)
  python -m benchmarks.run_benchmark -v                 # Verbose output
        """,
    )
    parser.add_argument(
        "--layer",
        choices=list(LAYER_MAP.keys()) + list(EXTERNAL_BENCHMARKS.keys()) + ["all"],
        default="all",
        help="Which benchmark layer to run (default: all)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose pytest output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSONL file",
    )
    parser.add_argument(
        "-k",
        type=str,
        default=None,
        help="pytest -k expression to filter tests",
    )

    args = parser.parse_args()

    # -- External benchmarks (standalone scripts, not pytest) --
    if args.layer in EXTERNAL_BENCHMARKS:
        script = EXTERNAL_BENCHMARKS[args.layer]
        print("=" * 60)
        print(f"  ENGRAM BENCHMARK -- {args.layer.upper()}")
        print(f"  Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        cmd = [sys.executable, "-m", f"benchmarks.{script.replace('.py', '')}"]
        result = subprocess.run(cmd, cwd=str(BENCHMARKS_DIR.parent))
        sys.exit(result.returncode)

    # -- Internal benchmarks (pytest-based) --
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]

    if args.layer == "all":
        cmd.append(str(BENCHMARKS_DIR))
    else:
        test_file = LAYER_MAP[args.layer]
        cmd.append(str(BENCHMARKS_DIR / test_file))

    # Always show print output
    cmd.append("-s")

    if args.verbose:
        cmd.append("-v")

    if args.k:
        cmd.extend(["-k", args.k])

    # Print header
    print("=" * 60)
    print(f"  ENGRAM BENCHMARK SUITE")
    print(f"  Layer: {args.layer}")
    print(f"  Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Run pytest
    result = subprocess.run(cmd, cwd=str(BENCHMARKS_DIR.parent))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
