"""Command-line entry point for the held-out inference demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pelgnn.inference import evaluate_sample


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Run PEL-GNN on eight held-out N03 minima."
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=root / "data/example.npz",
    )
    parser.add_argument(
        "--site-checkpoint",
        type=Path,
        default=root / "checkpoints/site_selector.pt",
    )
    parser.add_argument(
        "--mode-checkpoint",
        type=Path,
        default=root / "checkpoints/mode_proposer.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete machine-readable report",
    )
    return parser


def format_report(report: dict) -> str:
    aggregate = report["aggregate"]
    lines = [
        "PEL-GNN held-out inference demo",
        "",
        "Selection: first eight N03 minima",
        f"Minima: {report['minimum_count']}",
        f"Certified incident modes: {report['target_mode_count']}",
        (f"Mean best absolute mode overlap: {aggregate['mean_best_abs_overlap']:.5f}"),
        (
            "Median best absolute mode overlap: "
            f"{aggregate['median_best_abs_overlap']:.5f}"
        ),
        (
            "Targets with overlap >= 0.10: "
            f"{100 * aggregate['fraction_overlap_ge_0p10']:.1f}%"
        ),
        (
            "Mean site-selector top-10 activity uplift: "
            f"{aggregate['mean_site_top_10_uplift']:.3f}x"
        ),
        (f"Maximum proposal translation: {aggregate['max_proposal_translation']:.2e}"),
        (f"Maximum proposal norm error: {aggregate['max_proposal_norm_error']:.2e}"),
        "",
        (
            "This is a small inference example. "
            "The README table uses the grouped evaluation."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_sample(
        sample_path=args.sample,
        site_checkpoint=args.site_checkpoint,
        mode_checkpoint=args.mode_checkpoint,
        device_name=args.device,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()
