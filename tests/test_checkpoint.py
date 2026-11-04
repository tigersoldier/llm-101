import torch

from llm101.checkpoint import (
    load_adamw_state,
    load_model_artifact,
    load_payload,
    save_model_artifact,
    save_training_checkpoint,
)
from llm101.model import GPTConfig, TinyGPT
from llm101.optim import AdamW


def _small_model() -> TinyGPT:
    return TinyGPT(
        GPTConfig(
            vocab_size=16, d_model=8, n_layers=1, n_heads=2, d_ff=16, max_seq_len=8
        ),
        generator=torch.Generator().manual_seed(12),
    )


def test_model_artifact_round_trip_preserves_config_weights_and_metadata(
    tmp_path,
) -> None:
    model = _small_model()
    path = tmp_path / "model.pt"
    metadata = {"language": "English", "step": 7}
    save_model_artifact(path, model, metadata=metadata)

    loaded, loaded_metadata = load_model_artifact(path)

    assert loaded.config == model.config
    assert loaded_metadata == metadata
    for actual, expected in zip(loaded.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual.value, expected.value)


def test_training_checkpoint_preserves_optimizer_and_batch_rng(tmp_path) -> None:
    model = _small_model()
    optimizer = AdamW(model.parameters(), lr=0.0003)
    optimizer.parameters[0].grad.fill_(0.25)
    optimizer.step()
    generator = torch.Generator().manual_seed(99)
    torch.rand(5, generator=generator)
    path = tmp_path / "last.pt"

    save_training_checkpoint(
        path,
        model,
        optimizer,
        step=1,
        elapsed_seconds=12.5,
        tokens_seen=64,
        language_tokens_seen={"english": 64},
        batch_generator=generator,
        metadata={"run": "smoke"},
    )
    payload = load_payload(path)
    resumed_model, metadata = load_model_artifact(path)
    resumed_optimizer = AdamW(resumed_model.parameters(), lr=1.0)
    load_adamw_state(resumed_optimizer, payload["optimizer_state"])

    assert payload["training_state"]["step"] == 1
    assert payload["training_state"]["tokens_seen"] == 64
    assert metadata == {"run": "smoke"}
    assert resumed_optimizer.step_count == optimizer.step_count
    assert resumed_optimizer.lr == optimizer.lr
    torch.testing.assert_close(
        payload["training_state"]["batch_generator_state"], generator.get_state()
    )
    for actual, expected in zip(
        resumed_model.parameters(), model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual.value, expected.value)
