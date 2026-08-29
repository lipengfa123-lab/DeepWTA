"""项目路径引导与公共常量。

脚本/测试在 import 任何 deepwta / configs 模块之前，需要先将项目根目录与
src 目录加入 ``sys.path``。本模块集中管理这一样板逻辑，避免在每个脚本中
重复书写 4~5 行路径处理代码。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 本文件位于 <root>/src/deepwta/utils/env.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def add_project_paths() -> Path:
    """把项目根目录与 src 目录加入 sys.path，返回项目根路径。

    调用时机：模块 import deepwta.* 或 configs.* 之前（通常放在脚本开头）。
    """
    for path in (PROJECT_ROOT, SRC_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return PROJECT_ROOT
