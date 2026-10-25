import pytest
import torch
from torch.nn import functional as F

from llm_from_scratch.checkpoint import save_checkpoint
from llm_from_scratch.config import ModelConfig
from llm_from_scratch.model import GELU, GPT, LayerNorm


@pytest.fixture
def model():
    torch.manual_seed(42)
    return GPT(
        ModelConfig(context_length=8, embedding_dim=16, num_heads=2, num_layers=1, dropout=0.2)
    )


def test_explicit_components_match_pytorch():
    inputs = torch.randn(2, 4, 16)
    torch.testing.assert_close(LayerNorm(16)(inputs), F.layer_norm(inputs, (16,)))
    torch.testing.assert_close(GELU()(inputs), F.gelu(inputs, approximate="tanh"))


def test_causal_attention_cannot_see_future(model):
    model.eval()
    tokens = torch.randint(256, (2, 8))
    changed = tokens.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 256
    original, _ = model(tokens)
    modified, _ = model(changed)
    torch.testing.assert_close(original[:, :4], modified[:, :4], rtol=0, atol=0)
    assert not torch.equal(original[:, 4:], modified[:, 4:])


def test_loss_gradients_and_weight_tying(model):
    tokens = torch.randint(256, (2, 8))
    targets = torch.randint(256, (2, 8))
    logits, loss = model(tokens, targets)
    assert logits.shape == (2, 8, 256)
    torch.testing.assert_close(loss, F.cross_entropy(logits.reshape(-1, 256), targets.flatten()))
    assert model.lm_head.weight is model.token_embedding.weight
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_tiny_sequence_can_be_overfit():
    torch.manual_seed(1)
    model = GPT(
        ModelConfig(context_length=8, embedding_dim=16, num_heads=2, num_layers=1, dropout=0)
    )
    tokens = torch.arange(1, 9).unsqueeze(0)
    targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 1]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    _, initial = model(tokens, targets)
    for _ in range(80):
        _, loss = model(tokens, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    _, final = model(tokens, targets)
    assert final.item() < initial.item() * 0.1


def test_generation_crops_context_and_restores_training_mode(model):
    prompt = torch.arange(12).unsqueeze(0)
    greedy = model.generate(prompt, 6, temperature=0)
    top_one = model.generate(prompt, 6, top_k=1)
    assert greedy.shape == (1, 18)
    assert torch.equal(greedy[:, :12], prompt)
    assert torch.equal(greedy, top_one)
    assert model.training
    model.eval()
    model.generate(prompt, 1)
    assert not model.training


@pytest.mark.parametrize(
    "options",
    [
        {"max_new_tokens": -1},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"top_k": 0},
    ],
)
def test_generation_rejects_invalid_sampling(model, options):
    with pytest.raises(ValueError):
        model.generate(torch.ones((1, 1), dtype=torch.long), **options)


def test_invalid_input_shapes(model):
    with pytest.raises(ValueError, match="非空"):
        model(torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="超过"):
        model(torch.zeros((1, 9), dtype=torch.long))
    with pytest.raises(ValueError, match="形状"):
        model(torch.zeros((1, 2), dtype=torch.long), torch.zeros((1, 1), dtype=torch.long))


def test_weights_roundtrip_preserves_predictions(model, tmp_path):
    model.eval()
    tokens = torch.arange(8).unsqueeze(0)
    expected, _ = model(tokens)
    checkpoint = tmp_path / "weights.pt"
    save_checkpoint(checkpoint, {"model": model.state_dict()})
    restored = GPT(model.config)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["model"])
    restored.eval()
    actual, _ = restored(tokens)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
