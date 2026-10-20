"""Run the smoke or paper protocol and write the ledgers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clientlottery.battery import run_missing_battery  # noqa: E402
from clientlottery.simulate import run_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Client-lottery synthetic oracle checks")
    parser.add_argument(
        "--protocol",
        choices=("smoke", "paper", "paper_large", "smoke_missing", "paper_missing_battery"),
        required=True,
    )
    parser.add_argument("--T", type=int, default=None, help="Window length. Required for paper_large.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Defaults to results/ for smoke and results/paper_protocol/ for paper.",
    )
    parser.add_argument("--execution-host", default="local")
    args = parser.parse_args()
    if args.outdir is None:
        if args.protocol == "smoke":
            outdir = ROOT / "results"
        elif args.protocol == "paper_large":
            tag = args.T if args.T is not None else "unset"
            outdir = ROOT / "results" / f"paper_large_T{tag}"
        elif args.protocol == "smoke_missing":
            outdir = ROOT / "results" / "smoke_missing"
        elif args.protocol == "paper_missing_battery":
            tag = args.T if args.T is not None else "unset"
            outdir = ROOT / "results" / f"paper_missing_battery_T{tag}"
        else:
            outdir = ROOT / "results" / "paper_protocol"
    else:
        outdir = args.outdir
    seeds = tuple(args.seeds) if args.seeds is not None else None
    if args.protocol in ("smoke_missing", "paper_missing_battery"):
        summary = run_missing_battery(
            args.protocol,
            outdir,
            execution_host=args.execution_host,
            t_rounds=args.T,
            seeds=seeds,
        )
        checklist = summary.get("checklist", {})
        print(
            f"PROTOCOL_DONE name={summary['protocol']} paper_protocol={summary['paper_protocol']} "
            f"T={summary['T']} "
            f"A={checklist.get('A_dropout_itt_beats_naive')} "
            f"B={checklist.get('B_reuse_rows')} "
            f"C={checklist.get('C_shapley_loo_rows')} "
            f"D={checklist.get('D_surrogate_rows')} "
            f"E={checklist.get('E_drift_strong_path_signs')} "
            f"outdir={outdir}"
        )
        return
    summary = run_protocol(
        args.protocol,
        outdir,
        execution_host=args.execution_host,
        t_rounds=args.T,
        seeds=seeds,
    )
    win = summary.get("win_bar_evaluation", {})
    print(
        f"PROTOCOL_DONE name={summary['protocol']} paper_protocol={summary['paper_protocol']} "
        f"T={summary['run_definition']['T']} rows={summary['n_bernoulli_rows']} "
        f"win_bar_pass={win.get('win_bar_pass')} outdir={outdir}"
    )


if __name__ == "__main__":
    main()
