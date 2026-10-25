"""兼容 python generate.py；也可以使用安装后的 llm-generate 命令。"""

from llm_from_scratch.cli import generate_main as main

if __name__ == "__main__":
    main()
