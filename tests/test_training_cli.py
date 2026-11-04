import array
import json
import subprocess
import sys
from pathlib import Path

from llm101.checkpoint import load_payload
from llm101.tokenizer import BPETokenizer

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_token_stream(path: Path, token_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        array.array("I", token_ids).tofile(stream)


def test_training_cli_saves_artifacts_and_resumes_from_checkpoint(tmp_path) -> None:
    data_dir = tmp_path / "data"
    bilingual = data_dir / "bilingual"
    tokenizer = BPETokenizer.train(["hello small world", "你好，小世界"], vocab_size=32)
    tokenizer.save(data_dir / "tokenizers" / "bilingual.json")
    english_tokens = [index % tokenizer.vocab_size for index in range(128)]
    chinese_tokens = [(index + 3) % tokenizer.vocab_size for index in range(128)]
    _write_token_stream(bilingual / "train.bin", english_tokens + chinese_tokens)
    _write_token_stream(bilingual / "validation.bin", english_tokens + chinese_tokens)
    _write_token_stream(bilingual / "english_validation.bin", english_tokens)
    _write_token_stream(bilingual / "chinese_validation.bin", chinese_tokens)
    stats = {
        "vocab_size": tokenizer.vocab_size,
        "english_train_tokens": len(english_tokens),
        "chinese_train_tokens": len(chinese_tokens),
        "english_validation_tokens": len(english_tokens),
        "chinese_validation_tokens": len(chinese_tokens),
        "train_tokens": len(english_tokens) + len(chinese_tokens),
        "validation_tokens": len(english_tokens) + len(chinese_tokens),
    }
    manifest = {
        "english_revision": "smoke-test",
        "chinese_revision": "smoke-test",
        "source_file_sha256": {},
        "chinese_validation_fraction": 0.01,
        "tokenizer_training": {},
        "corpora": {"bilingual": stats},
        "tokenizer_files": {"bilingual": "tokenizers/bilingual.json"},
    }
    (data_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    run_dir = tmp_path / "run"

    def run(hours: str, *, resume: bool = False) -> None:
        command = [
            sys.executable,
            "scripts/train_model.py",
            "--hours",
            hours,
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(run_dir),
            "--batch-size",
            "2",
            "--sequence-length",
            "4",
            "--d-model",
            "8",
            "--layers",
            "1",
            "--heads",
            "2",
            "--d-ff",
            "16",
            "--eval-batches",
            "1",
            "--eval-every-seconds",
            "100",
            "--checkpoint-every-seconds",
            "100",
            "--log-every-steps",
            "1",
            "--threads",
            "1",
        ]
        if resume:
            command.append("--resume")
        subprocess.run(
            command,
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

    run("0.0001")
    checkpoint_path = run_dir / "last.pt"
    first_payload = load_payload(checkpoint_path)
    first_step = first_payload["training_state"]["step"]
    assert first_step > 0
    assert (run_dir / "model-0.0001h.pt").is_file()
    samples = json.loads((run_dir / "samples-0.0001h.json").read_text("utf-8"))
    assert len(samples["samples"]) == 4
    assert (run_dir / "run_manifest.json").is_file()

    run("0.0005", resume=True)
    resumed_payload = load_payload(checkpoint_path)
    state = resumed_payload["training_state"]
    assert state["step"] > first_step
    assert set(state["language_tokens_seen"]) == {"english", "chinese"}
    assert sum(state["language_tokens_seen"].values()) == state["tokens_seen"]
    assert (run_dir / "model-0.0005h.pt").is_file()
