"""Diagnostic extension of the legacy return-value proxy; not a Python executor.

Normalize name-only augmented assignments to ordinary arithmetic assignments.
The legacy numeric/resource bounds and refusals still apply. In particular,
print arguments remain ignored and control flow is unsupported. This module is
separate so frozen metrics and production selectors retain their exact behavior.
"""
from __future__ import annotations

import ast

from .arithmetic import Evaluation, MAX_AST_NODES, MAX_SOURCE_BYTES, evaluate_code


class _AugAssign(ast.NodeTransformer):
    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        if not isinstance(node.target, ast.Name):
            return node
        return ast.copy_location(ast.Assign(
            targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
            value=ast.BinOp(left=ast.Name(id=node.target.id, ctx=ast.Load()),
                            op=node.op, right=node.value)), node)


def evaluate_code_augassign(source: str) -> Evaluation:
    """Keep legacy successes; interpret numeric ``x op= y`` via ``x = x op y``."""
    original = evaluate_code(source)
    if original.supported or not isinstance(source, str):
        return original
    if len(source.encode('utf-8', errors='replace')) > MAX_SOURCE_BYTES:
        return original
    try:
        tree = ast.parse(source)
        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_AST_NODES or not any(isinstance(n, ast.AugAssign) for n in nodes):
            return original
        normalized = ast.fix_missing_locations(_AugAssign().visit(tree))
        return evaluate_code(ast.unparse(normalized))
    except (SyntaxError, RecursionError, MemoryError, ValueError):
        return original
