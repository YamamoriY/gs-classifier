"""CLI のテスト"""

import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "gs_classifier.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ground" in result.stdout
    assert "trunk" in result.stdout
    assert "tree" in result.stdout
    assert "identify" in result.stdout
    assert "view" in result.stdout


def test_cli_no_command_shows_help():
    result = subprocess.run(
        [sys.executable, "-m", "gs_classifier.cli"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "ground" in result.stderr or "ground" in result.stdout


def test_cli_ground_help():
    result = subprocess.run(
        [sys.executable, "-m", "gs_classifier.cli", "ground", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--ply" in result.stdout
    assert "--no-viewer" in result.stdout


def test_cli_identify_help():
    result = subprocess.run(
        [sys.executable, "-m", "gs_classifier.cli", "identify", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "image" in result.stdout
    assert "--no-display" in result.stdout


def test_cli_view_help():
    result = subprocess.run(
        [sys.executable, "-m", "gs_classifier.cli", "view", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "files" in result.stdout
