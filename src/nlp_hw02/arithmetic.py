"""A deliberately small, non-executing interpreter for numeric ``solve`` bodies.

This is a development metric helper, not a Python sandbox and not the course's
official pass@1 evaluator. Unsupported syntax returns ``supported=False``.
Generated source is parsed into an AST and interpreted node by node; it is never
compiled, imported, evaluated, or executed.
"""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass
from typing import Any


MAX_SOURCE_BYTES = 65_536
MAX_AST_NODES = 4_096
MAX_STATEMENTS = 512
MAX_LOCALS = 256
MAX_INT_BITS = 256
MAX_ABS_VALUE = 1e100
MAX_POWER_EXPONENT = 12
MAX_ROUND_DIGITS = 12


@dataclass(frozen=True)
class Evaluation:
    """Result of evaluating a supported numeric return, or a refusal reason."""

    supported: bool
    value: int | float | None = None
    reason: str | None = None


class _Unsupported(Exception):
    pass


def _bounded_number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _Unsupported("non-numeric value")
    if isinstance(value, int):
        if value.bit_length() > MAX_INT_BITS:
            raise _Unsupported("integer bit limit")
        if abs(value) > MAX_ABS_VALUE:
            raise _Unsupported("magnitude limit")
    elif not math.isfinite(value) or abs(value) > MAX_ABS_VALUE:
        raise _Unsupported("non-finite or excessive magnitude")
    return value


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CALLS = {"int": int, "float": float, "round": round, "abs": abs,
          "min": min, "max": max}


class _Interpreter:
    def __init__(self) -> None:
        self.locals: dict[str, int | float] = {}

    def expr(self, node: ast.expr) -> int | float:
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise _Unsupported("only numeric literals")
            return _bounded_number(node.value)

        if isinstance(node, ast.Name):
            if node.id not in self.locals:
                raise _Unsupported("unknown variable")
            return self.locals[node.id]

        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _bounded_number(_UNARY[type(node.op)](self.expr(node.operand)))

        if isinstance(node, ast.BinOp):
            left, right = self.expr(node.left), self.expr(node.right)
            if isinstance(node.op, ast.Pow):
                if type(right) not in (int, float) or int(right) != right:
                    raise _Unsupported("power must have integer exponent")
                if abs(right) > MAX_POWER_EXPONENT:
                    raise _Unsupported("power exponent limit")
                fn = operator.pow
            else:
                fn = _BINARY.get(type(node.op))
                if fn is None:
                    raise _Unsupported("binary operator")
            try:
                return _bounded_number(fn(left, right))
            except (ArithmeticError, OverflowError, ValueError) as exc:
                raise _Unsupported("arithmetic error") from exc

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _CALLS.get(node.func.id)
            if fn is None or node.keywords:
                raise _Unsupported("call not whitelisted")
            if not node.args:
                raise _Unsupported("empty numeric call")
            args = [self.expr(arg) for arg in node.args]
            if fn is round:
                if len(args) not in (1, 2):
                    raise _Unsupported("round arity")
                if len(args) == 2 and (type(args[1]) is not int or
                                       abs(args[1]) > MAX_ROUND_DIGITS):
                    raise _Unsupported("round digit limit")
            elif fn in (int, float, abs) and len(args) != 1:
                raise _Unsupported("numeric call arity")
            elif fn in (min, max) and len(args) < 1:
                raise _Unsupported("numeric call arity")
            try:
                return _bounded_number(fn(*args))
            except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
                raise _Unsupported("numeric call error") from exc

        raise _Unsupported("expression syntax")

    def run(self, function: ast.FunctionDef) -> int | float:
        if function.args.args or function.args.posonlyargs or function.args.kwonlyargs:
            raise _Unsupported("solve must take no arguments")
        if function.args.vararg or function.args.kwarg or function.decorator_list:
            raise _Unsupported("function features")

        statements = 0
        for statement in function.body:
            statements += 1
            if statements > MAX_STATEMENTS:
                raise _Unsupported("statement limit")
            if (isinstance(statement, ast.Expr) and
                    isinstance(statement.value, ast.Constant) and
                    isinstance(statement.value.value, str)):
                continue  # docstring
            if (isinstance(statement, ast.Expr) and
                    isinstance(statement.value, ast.Call) and
                    isinstance(statement.value.func, ast.Name) and
                    statement.value.func.id == "print"):
                continue  # print output/arguments are outside this return-value proxy
            if isinstance(statement, ast.Assign):
                if not statement.targets or any(
                    not isinstance(target, ast.Name) for target in statement.targets
                ):
                    raise _Unsupported("assignment target")
                if any(target.id in {*_CALLS, "print"} for target in statement.targets):
                    raise _Unsupported("shadowed whitelisted name")
                value = self.expr(statement.value)
                for target in statement.targets:
                    self.locals[target.id] = value
            elif isinstance(statement, ast.AnnAssign):
                if not isinstance(statement.target, ast.Name) or statement.value is None:
                    raise _Unsupported("annotated assignment")
                if statement.target.id in {*_CALLS, "print"}:
                    raise _Unsupported("shadowed whitelisted name")
                self.locals[statement.target.id] = self.expr(statement.value)
            elif isinstance(statement, ast.Return):
                if statement.value is None:
                    raise _Unsupported("empty return")
                return self.expr(statement.value)
            else:
                raise _Unsupported("statement syntax")
            if len(self.locals) > MAX_LOCALS:
                raise _Unsupported("local variable limit")
        raise _Unsupported("no top-level numeric return")


def evaluate_code(source: str) -> Evaluation:
    """Interpret a restricted, straight-line ``def solve(): return number``.

    Allowed: numeric literals, local-name assignments, unary +/-; +, -, *, /,
    //, %, and bounded integer powers; and calls to int/float/round/abs/min/max.
    Standalone ``print(...)`` statements and their arguments are ignored. A
    docstring is ignored. This means print-time exceptions are outside this
    return-value proxy. Control
    flow, imports, attributes, indexing, comprehensions, loops, unknown calls,
    and all other Python syntax are rejected. Resource bounds are constants at
    module scope; numeric results are finite and capped.
    """

    if not isinstance(source, str):
        return Evaluation(False, reason="source is not text")
    if len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        return Evaluation(False, reason="source byte limit")
    try:
        tree = ast.parse(source, mode="exec")
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            raise _Unsupported("AST node limit")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise _Unsupported("expected one top-level solve function")
        function = tree.body[0]
        if function.name != "solve":
            raise _Unsupported("expected solve function")
        value = _Interpreter().run(function)
        return Evaluation(True, value=value)
    except (SyntaxError, RecursionError, MemoryError, ValueError):
        return Evaluation(False, reason="parse failure or resource limit")
    except _Unsupported as exc:
        return Evaluation(False, reason=str(exc))
