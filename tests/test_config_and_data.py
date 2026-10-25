import json

import pytest
import torch

from llm_from_scratch.config import ExperimentConfig, ModelConfig, TrainingConfig
from llm_from_scratch.dataset import get_batch, load_data
from llm_from_scratch.tokenizer import ByteTokenizer


def test_unicode_tokenizer_roundtrip():
    tokenizer = ByteTokenizer()
    text = "你好，GPT! 🧠\n"
    encoded = tokenizer.encode(text)
    assert tokenizer.decode(encoded, errors="strict") == text
    assert all(0 <= token < tokenizer.vocab_size for token in encoded)
    assert tokenizer.decode([0xE4]) == "\ufffd"


def test_config_roundtrip_and_example_defaults(tmp_path):
    config = ExperimentConfig()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert ExperimentConfig.load(path) == config
    assert ExperimentConfig.load("configs/tiny.json") == config


@pytest.mark.parametrize(
    "values",
    [
        {"num_heads": 3},
        {"context_length": 0},
        {"num_layers": -1},
        {"embedding_dim": True},
        {"dropout": 1},
        {"dropout": float("nan")},
    ],
)
def test_invalid_model_config(values):
    with pytest.raises(ValueError):
        ModelConfig(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"batch_size": 0},
        {"max_steps": 1.5},
        {"train_fraction": 1},
        {"learning_rate": 0},
        {"seed": -1},
        {"eval_batches": 0},
        {"device": "invalid"},
        {"output_dir": ""},
        {"data_path": 1},
    ],
)
def test_invalid_training_config(values):
    with pytest.raises(ValueError):
        TrainingConfig(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"typo": {}},
        {"model": {"num_head": 4}},
        {"training": []},
        {"model": {"vocab_size": 100}},
        [],
    ],
)
def test_invalid_experiment_config(values):
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(values)


def test_split_before_sampling_and_shift_once(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a" * 90 + "b" * 10, encoding="utf-8")
    train, val = load_data(corpus)
    assert train.tolist() == [ord("a")] * 90
    assert val.tolist() == [ord("b")] * 10
    inputs, targets = get_batch(torch.arange(20), 4, 8, "cpu")
    assert inputs.shape == targets.shape == (4, 8)
    torch.testing.assert_close(targets, inputs + 1)


def test_shortest_batch_and_reproducible_sampling():
    inputs, targets = get_batch(torch.arange(9), 2, 8, "cpu")
    assert inputs[0].tolist() == list(range(8))
    assert targets[0].tolist() == list(range(1, 9))
    first, _ = get_batch(torch.arange(100), 8, 8, "cpu", generator=torch.Generator().manual_seed(7))
    second, _ = get_batch(
        torch.arange(100), 8, 8, "cpu", generator=torch.Generator().manual_seed(7)
    )
    assert torch.equal(first, second)
    with pytest.raises(ValueError, match="至少需要"):
        get_batch(torch.arange(8), 2, 8, "cpu")


def test_bundled_corpus():
    train, val = load_data()
    assert len(train) > len(val) > 64
