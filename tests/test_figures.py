"""T7: the low-fi vs high-fi figure script runs and writes an image (visual check is manual)."""

import subprocess
import sys

import pytest
from conftest import ROOT


@pytest.mark.slow
def test_t7_figure_script(tmp_path):
    out = tmp_path / "hifi_vs_lowfi.png"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_hifi_vs_lowfi.py"), str(out), "--n", "2"],
                   check=True, capture_output=True, timeout=900)
    assert out.exists() and out.stat().st_size > 50_000
