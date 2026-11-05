"""End-to-end regression test on synthetic data.

The synthetic generator plants a known effect (Arab odds ratio ~0.55 on the corporate
share). The pipeline must recover a negative, significant coefficient and pass every
invariant in selftest.py.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def run(script, cwd):
    result = subprocess.run(
        [sys.executable, script], cwd=cwd, capture_output=True, text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, f"{script} failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    return result.stdout


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    work = tmp_path_factory.mktemp("pipeline")
    for f in SRC.glob("*.py"):
        shutil.copy(f, work)
    for script in ("make_synthetic.py", "build_panel.py", "analyze.py"):
        run(script, work)
    return work


def test_panel_built(workdir):
    panel = pd.read_csv(workdir / "panel.csv")
    assert len(panel) > 1000
    assert (workdir / "flow.csv").exists()


def test_planted_effect_recovered(workdir):
    ladder = pd.read_csv(workdir / "results_ladder.csv")
    main = ladder[ladder["model"].str.contains("BINOMIAL")]
    assert len(main) > 0
    first = main.iloc[0]
    assert first["beta_arab"] < 0, "pipeline did not recover the planted negative effect"
    assert first["p"] < 0.05, "planted effect recovered but not significant"


def test_invariants(workdir):
    out = run("selftest.py", workdir)
    assert "0 failed" in out
