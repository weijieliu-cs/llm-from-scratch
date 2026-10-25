"""固定的 UTF-8 字节词表，无需训练或下载外部 Tokenizer。"""

from collections.abc import Iterable


class ByteTokenizer:
    vocab_size = 256
    name = "utf8-bytes-v1"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Iterable[int], *, errors: str = "replace") -> str:
        """生成可能截断 UTF-8 字符；默认用替换字符显示无效序列。"""
        return bytes(token_ids).decode("utf-8", errors=errors)
