import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def small_cpu_models():
    torch.set_num_threads(1)
