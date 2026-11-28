"""Tests for the stage 1–3 command-line reports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _PROJECT_ROOT / "scripts" / "show_tokenizer.py"


def _run_stage(stage: int) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PROJECT_ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--stage", str(stage)],
        cwd=_PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_english_baseline_overflows_context() -> None:
    output = _run_stage(1)

    assert "text            the cat sat on the mat" in output
    assert "chars           22" in output
    assert "ids (24)" in output
    assert "unknown         0" in output
    assert "hand merge      13 symbols; + specials = 15" in output
    assert "ValueError: sequence length 24 exceeds the model context of 16" in output


def test_english_hand_merge_shares_at() -> None:
    output = _run_stage(1)

    assert (
        "symbols         ['the', ' ', 'c', 'at', ' ', 'sat', ' ', 'o', 'n', "
        "' ', 'the', ' ', 'mat']"
    ) in output


def test_chinese_baseline_is_unknown() -> None:
    output = _run_stage(1)

    assert "text            小猫坐在垫子上" in output
    assert "chars           7" in output
    assert "ids (9)" in output
    assert "unknown         7" in output
    assert "decoded         ''" in output
    assert "model           -" in output


def test_stage_two_learns_and_applies_merges() -> None:
    output = _run_stage(2)

    assert "stage 2 | adjacent-pair merges (16)" in output
    assert "'a' + 't' -> 'at'" in output
    assert "symbols (5)" in output
    assert "symbols (2)" in output


def test_stage_three_trains_teaching_vocabulary() -> None:
    output = _run_stage(3)

    assert "stage 3 | vocabulary" in output
    assert "vocab size      512" in output
    assert "alphabet        143" in output
    assert "merges          366" in output
    assert "ids (4)" in output
    assert "ids (3)" in output
