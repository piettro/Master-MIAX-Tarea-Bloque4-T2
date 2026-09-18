"""End-to-end run of the whole pipeline on small synthetic data."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from main import main
from src.pipeline import run_pipeline
from src.utils import config
from src.utils.config import RunSettings


@pytest.mark.filterwarnings("ignore")
def test_pipeline_end_to_end(data_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    settings = replace(
        RunSettings(
            data_dir=data_dir,
            output_dir=output_dir,
            production_output_dir=output_dir / "delivery",
            include_xgboost=False,
            show_progress=False,
        ).quick(),
        random_forest_trees=10,
        bandit_steps=500,
        random_benchmark_steps=200,
        shap_sample_size=100,
        dependence_sample_size=150,
        bootstrap_samples=50,
        counterfactual_candidates=200,
    )
    results = run_pipeline(settings)

    production = pd.read_csv(data_dir / config.PRODUCTION_FILENAME)
    for filename in config.SCENARIO_OUTPUT_FILENAMES.values():
        delivered = pd.read_csv(output_dir / "delivery" / filename)
        assert list(delivered.columns) == [config.PREDICTION_COLUMN]
        assert len(delivered) == len(production)
        assert set(delivered[config.PREDICTION_COLUMN]) <= {0, 1}

    files = results["production"]["files"]
    # A 1:10 cost ratio must deny at least as often as a 1:1 ratio.
    assert (files[config.SCENARIO_2_KEY]["denial_rate"]
            >= files[config.SCENARIO_1_KEY]["denial_rate"])
    assert (output_dir / "report.md").is_file()
    assert json.loads((output_dir / "reports" / "results.json").read_text(
        encoding="utf-8"))["model_selection"]["winner"]
    assert any((output_dir / "figures").glob("*.png"))


def test_main_reports_missing_data(tmp_path: Path) -> None:
    exit_code = main(["--data-dir", str(tmp_path), "--output-dir",
                      str(tmp_path / "out"), "--no-progress"])
    assert exit_code == 1
