"""Tests for the section 2.1 stage-1 command-line report."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _PROJECT_ROOT / "scripts" / "show_tokenizer.py"


def _run_stage_one() -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PROJECT_ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--stage", "1"],
        cwd=_PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_english_baseline_overflows_context() -> None:
    output = _run_stage_one()

    assert "text            the cat sat on the mat" in output
    assert "chars           22" in output
    assert "ids (24)" in output
    assert "unknown         0" in output
    assert "hand merge      13 symbols; + specials = 15" in output
    assert "ValueError: sequence length 24 exceeds the model context of 16" in output


def test_english_hand_merge_shares_at() -> None:
    output = _run_stage_one()

    assert (
        "symbols         ['the', ' ', 'c', 'at', ' ', 'sat', ' ', 'o', 'n', "
        "' ', 'the', ' ', 'mat']"
    ) in output


def test_chinese_baseline_is_unknown() -> None:
    output = _run_stage_one()

    assert "text            小猫坐在垫子上" in output
    assert "chars           7" in output
    assert "ids (9)" in output
    assert "unknown         7" in output
    assert "decoded         ''" in output
    assert "model           -" in output
