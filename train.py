"""运行一个最小 GPT 训练示例，观察 loss 并生成文本。"""

from pathlib import Path

import torch

from dataset import get_batch, load_data
from model import GPT, ModelConfig


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, num_batches):
    model.eval()
    device = next(model.parameters()).device
    losses = []
    for tokens in (train_data, val_data):
        total = 0.0
        for _ in range(num_batches):
            inputs, targets = get_batch(tokens, batch_size, model.config.context_length, device)
            _, loss = model(inputs, targets)
            total += loss.item()
        losses.append(total / num_batches)
    model.train()
    return losses


def main():
    # 训练参数直接在这里修改。
    batch_size = 8
    max_steps = 200
    learning_rate = 1e-3
    eval_interval = 50
    eval_batches = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1337)
    torch.set_num_threads(1)  # 小模型在 CPU 上使用少量线程即可。
    data_path = Path(__file__).parent / "data" / "sample.txt"
    train_data, val_data = load_data(data_path)
    model = GPT(ModelConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"device={device}, parameters={sum(p.numel() for p in model.parameters()):,}")

    for step in range(1, max_steps + 1):
        inputs, targets = get_batch(train_data, batch_size, model.config.context_length, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % eval_interval == 0 or step == max_steps:
            train_loss, val_loss = estimate_loss(
                model, train_data, val_data, batch_size, eval_batches
            )
            print(f"step={step:4d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    # 自回归生成：每次采样一个新 token，再把它接到输入末尾。
    model.eval()
    prompt = "The workshop"
    tokens = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    generated = model.generate(tokens, max_new_tokens=200)
    print("\n" + bytes(generated[0].tolist()).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
