"""Command-line entry point of the explainable credit-scoring project.

Usage::

    python main.py                 # full run (regenerates delivery files)
    python main.py --quick         # lighter smoke run
    python main.py --help          # all options
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.pipeline import run_pipeline
from src.utils import config
from src.utils.config import RunSettings
from src.utils.logging_config import configure_logging

logger = logging.getLogger("main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build, optimise and audit (XAI) a credit-scoring "
                    "model under two cost scenarios.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR,
                        help="folder with the input CSV files")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR,
                        help="folder for figures and reports")
    parser.add_argument("--production-output-dir", type=Path,
                        default=config.DATA_DIR,
                        help="folder for cs_produccion1/2.csv")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="global random seed")
    parser.add_argument("--quick", action="store_true",
                        help="smaller samples / fewer folds (smoke run)")
    parser.add_argument("--tune", action="store_true",
                        help="rerun the HistGB randomised search")
    parser.add_argument("--no-xgboost", action="store_true",
                        help="exclude XGBoost from the candidates")
    parser.add_argument("--no-progress", action="store_true",
                        help="hide progress bars")
    parser.add_argument("--log-level", default="INFO",
                        help="DEBUG, INFO, WARNING or ERROR")
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> RunSettings:
    """Translate CLI arguments into :class:`RunSettings`.

    Args:
        args: Parsed arguments.

    Returns:
        Runtime settings.
    """
    settings = RunSettings(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        production_output_dir=args.production_output_dir,
        seed=args.seed,
        tune=args.tune,
        include_xgboost=not args.no_xgboost,
        show_progress=not args.no_progress,
    )
    return settings.quick() if args.quick else settings


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    configure_logging(args.log_level, args.output_dir / "run.log")
    settings = build_settings(args)
    start = time.perf_counter()
    try:
        results = run_pipeline(settings)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    for key, info in results["production"]["files"].items():
        logger.info("%s -> %s (%.2f%% denied)", config.SCENARIO_LABELS[key],
                    info["path"], 100 * info["denial_rate"])
    logger.info("Done in %.1f s. Report: %s", time.perf_counter() - start,
                settings.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
