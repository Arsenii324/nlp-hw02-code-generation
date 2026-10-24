"""Focused checks for the bounded arithmetic interpreter (no external data)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlp_hw02.arithmetic import evaluate_code


def check(source: str, expected: int | float | None, supported: bool = True) -> None:
    result = evaluate_code(source)
    assert result.supported is supported, (source, result)
    if supported:
        assert result.value == expected, (source, result.value, expected)


def main() -> None:
    check("def solve():\n    return 2 + 3 * 4\n", 14)
    check("def solve():\n    x = 17 // 5\n    return x % 3\n", 0)
    check("def solve():\n    print('ignored')\n    return round(2.25, 1)\n", 2.2)
    check("def solve():\n    return -2 ** 3\n", -8)

    check("def solve():\n    return 1 / 0\n", None, False)
    check("def solve():\n    return 2 ** 13\n", None, False)
    check("def solve():\n    return min(1)\n", None, False)
    check("def solve():\n    import os\n    return 1\n", None, False)
    check("def solve():\n    return __import__('os')\n", None, False)
    check("def solve():\n    return (1).__class__\n", None, False)
    check("def solve():\n    return x\n", None, False)
    check("def solve():\n    int = 3\n    return int(2)\n", None, False)
    check("def solve():\n    print(1 / 0)\n    return 7\n", 7)
    check("def solve():\n    return 10 ** 1000\n", None, False)
    check("def solve():\n    return " + "9" * 5000 + "\n", None, False)
    check("def solve():\n    return 1\x00\n", None, False)
    check("def solve():\n    while True:\n        return 1\n", None, False)

    overlong = "def solve():\n    return " + "1+" * 5000 + "1\n"
    check(overlong, None, False)
    print("arithmetic checks: 18 passed")


if __name__ == "__main__":
    main()
