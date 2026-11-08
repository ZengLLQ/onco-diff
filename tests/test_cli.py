"""Unit tests for OncoDiff CLI."""

import subprocess
import sys


def test_cli_help():
    res = subprocess.run([sys.executable, "-m", "onco_diff.cli", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "OncoDiff" in res.stdout


def test_cli_simulate():
    res = subprocess.run(
        [sys.executable, "-m", "onco_diff.cli", "simulate", "--days", "10.0"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Spatiotemporal Glioblastoma Neuro-Oncology Report" in res.stdout
    assert "T1Gd Enhancing Core" in res.stdout


def test_cli_stupp():
    res = subprocess.run(
        [sys.executable, "-m", "onco_diff.cli", "stupp", "--days", "15.0", "--mgmt-methylated"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Spatiotemporal Glioblastoma Neuro-Oncology Report" in res.stdout


def test_cli_prognosis():
    res = subprocess.run(
        [sys.executable, "-m", "onco_diff.cli", "prognosis", "--diff", "0.18", "--rho", "0.025"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Clinical Invasiveness and Survival Prognosis" in res.stdout
    assert "Swanson Invasiveness Ratio" in res.stdout
