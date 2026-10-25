import json
from dataclasses import replace

import pytest
import torch

from llm_from_scratch.checkpoint import load_checkpoint
from llm_from_scratch.cli import generate_text
from llm_from_scratch.config import ExperimentConfig, ModelConfig, TrainingConfig
from llm_from_scratch.dataset import load_data
from llm_from_scratch.model import GPT
from llm_from_scratch.training import create_optimizer, estimate_loss, train


def experiment(output_dir, steps=4):
    return ExperimentConfig(
        model=ModelConfig(
            context_length=8, embedding_dim=16, num_heads=2, num_layers=1, dropout=0.2
        ),
        training=TrainingConfig(
            output_dir=str(output_dir),
            max_steps=steps,
            batch_size=2,
            eval_batches=2,
            eval_interval=3,
            device="cpu",
        ),
    )


def test_evaluation_is_repeatable_and_preserves_rng_and_mode():
    model = GPT(experiment("unused").model)
    train_data, val_data = load_data()
    original_rng = torch.get_rng_state()
    first = estimate_loss(model, train_data, val_data, 2, 2)
    assert model.training
    assert torch.equal(torch.get_rng_state(), original_rng)
    model.eval()
    assert estimate_loss(model, train_data, val_data, 2, 2) == first
    assert not model.training


def test_optimizer_groups_shared_weights_once():
    model = GPT(experiment("unused").model)
    optimizer = create_optimizer(model, 1e-3, 0.01)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters}) == len(list(model.parameters()))
    assert all(p.ndim >= 2 for p in optimizer.param_groups[0]["params"])
    assert optimizer.param_groups[1]["weight_decay"] == 0


def test_resume_matches_uninterrupted_training_exactly(tmp_path):
    complete = load_checkpoint(train(experiment(tmp_path / "complete", steps=6)))
    config = experiment(tmp_path / "resumed", steps=2)
    checkpoint = train(config)
    # 中断点不在原评估周期上；改变评估频率仍不能改变训练随机数流。
    continued = replace(
        config, training=replace(config.training, max_steps=6, eval_interval=1, eval_batches=1)
    )
    resumed = load_checkpoint(train(continued, resume=checkpoint))
    assert complete["step"] == resumed["step"] == 6
    for name, weight in complete["model"].items():
        assert torch.equal(weight, resumed["model"][name]), name
    for key in ("batch_rng_state", "torch_rng_state"):
        assert torch.equal(complete[key], resumed[key])
    for index, values in complete["optimizer"]["state"].items():
        for key, value in values.items():
            assert torch.equal(value, resumed["optimizer"]["state"][index][key])


def test_checkpoint_generation_and_metrics(tmp_path):
    config = experiment(tmp_path)
    checkpoint = train(config)
    assert (tmp_path / "best.pt").exists()
    assert ExperimentConfig.load(tmp_path / "config.json").model == config.model
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [1, 3, 4]
    assert records[-1]["tokens_seen"] == 4 * 2 * 8
    assert generate_text(checkpoint, "你好", max_new_tokens=2, temperature=0).startswith("你好")
    assert generate_text(checkpoint, "Hi", max_new_tokens=0) == "Hi"


def test_resume_rejects_changed_data_and_settings(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefghij" * 100, encoding="utf-8")
    config = experiment(tmp_path / "run", steps=1)
    config = replace(config, training=replace(config.training, data_path=str(corpus)))
    checkpoint = train(config)
    continued = replace(config, training=replace(config.training, max_steps=2))
    changed = replace(continued, training=replace(continued.training, learning_rate=0.1))
    with pytest.raises(ValueError, match="learning_rate"):
        train(changed, resume=checkpoint)
    corpus.write_text("jihgfedcba" * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="校验值变化"):
        train(continued, resume=checkpoint)


def test_existing_experiment_is_not_overwritten(tmp_path):
    config = experiment(tmp_path, steps=1)
    checkpoint = train(config)
    original = checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        train(config)
    assert checkpoint.read_bytes() == original


def test_resume_from_other_snapshot_requires_new_directory(tmp_path):
    config = experiment(tmp_path / "source", steps=1)
    train(config)
    continued = replace(config, training=replace(config.training, max_steps=2))
    with pytest.raises(ValueError, match="其他快照"):
        train(continued, resume=tmp_path / "source" / "best.pt")
    target = tmp_path / "target"
    fork = replace(continued, training=replace(continued.training, output_dir=str(target)))
    assert train(fork, resume=tmp_path / "source" / "best.pt") == target / "last.pt"
