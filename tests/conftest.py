"""pytest 全局路径引导。

把项目根目录与 src 加入 sys.path，使测试可以直接导入
``configs.*`` 与 ``deepwta.*``，无需每个测试文件重复样板代码。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
