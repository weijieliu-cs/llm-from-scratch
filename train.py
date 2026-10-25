"""兼容 python train.py；完整命令行帮助见 python train.py --help。"""

from llm_from_scratch.cli import train_main as main
from llm_from_scratch.training import estimate_loss

__all__ = ["estimate_loss", "main"]

if __name__ == "__main__":
    main()
