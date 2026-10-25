import subprocess
import sys
from pathlib import Path

from llm_from_scratch.cli import generate_main, train_main


def test_training_generation_and_resume_cli(tmp_path, capsys):
    train_main(["--steps", "1", "--device", "cpu", "--output-dir", str(tmp_path)])
    generate_main(
        [
            "--checkpoint",
            str(tmp_path / "last.pt"),
            "--prompt",
            "The workshop",
            "--max-new-tokens",
            "2",
            "--temperature",
            "0",
        ]
    )
    train_main(["--resume", str(tmp_path / "last.pt"), "--steps", "2"])
    output = capsys.readouterr().out
    assert "The workshop" in output
    assert "step=   2" in output


def test_legacy_entry_points(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for script in ("train.py", "generate.py"):
        result = subprocess.run(
            [sys.executable, str(root / script), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "usage:" in result.stdout
