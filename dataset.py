"""兼容原有导入；数据实现位于 llm_from_scratch.dataset。"""

from llm_from_scratch.dataset import get_batch, load_data

__all__ = ["get_batch", "load_data"]
