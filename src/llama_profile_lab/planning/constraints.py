"""Restricted expression parser for search constraints and conditions."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any


class ConstraintError(ValueError):
    """Raised for invalid or unevaluable search expressions."""


def _parse(expression: str) -> ast.Expression:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ConstraintError(f"invalid constraint syntax: {expression}") from exc
    if not isinstance(tree, ast.Expression):
        raise ConstraintError("constraint must be an expression")
    _validate_node(tree.body)
    return tree


def validate_constraint(expression: str) -> None:
    """Validate syntax without evaluating against a Candidate."""
    _parse(expression)


def evaluate_constraint(expression: str, candidate: Mapping[str, Any]) -> bool:
    """Evaluate a restricted expression against a Candidate payload."""
    tree = _parse(expression)
    result = _evaluate_node(tree.body, candidate)
    if not isinstance(result, bool):
        raise ConstraintError(f"constraint did not evaluate to bool: {expression}")
    return result


def _validate_node(node: ast.AST) -> None:
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ConstraintError("only and/or boolean operators are supported")
        for value in node.values:
            _validate_node(value)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.Not):
            raise ConstraintError("only boolean not is supported")
        _validate_node(node.operand)
        return

    if isinstance(node, ast.Compare):
        _validate_node(node.left)
        for operator in node.ops:
            if not isinstance(
                operator,
                (
                    ast.Eq,
                    ast.NotEq,
                    ast.Lt,
                    ast.LtE,
                    ast.Gt,
                    ast.GtE,
                    ast.In,
                    ast.NotIn,
                ),
            ):
                raise ConstraintError("unsupported comparison operator")
        for comparator in node.comparators:
            _validate_node(comparator)
        return

    if isinstance(node, ast.Attribute):
        _validate_node(node.value)
        return

    if isinstance(node, ast.Name | ast.Constant):
        return

    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        for element in node.elts:
            _validate_node(element)
        return

    raise ConstraintError(
        f"unsupported constraint syntax: {node.__class__.__name__}"
    )


def _evaluate_node(node: ast.AST, candidate: Mapping[str, Any]) -> object:
    if isinstance(node, ast.BoolOp):
        values = [_as_bool(_evaluate_node(value, candidate)) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        return any(values)

    if isinstance(node, ast.UnaryOp):
        return not _as_bool(_evaluate_node(node.operand, candidate))

    if isinstance(node, ast.Compare):
        left = _evaluate_node(node.left, candidate)
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = _evaluate_node(comparator, candidate)
            if not _compare(left, operator, right):
                return False
            left = right
        return True

    if isinstance(node, ast.Attribute):
        parent = _evaluate_node(node.value, candidate)
        if not isinstance(parent, Mapping):
            raise ConstraintError(
                f"cannot resolve .{node.attr} on non-object value"
            )
        if node.attr not in parent:
            raise ConstraintError(f"unknown Candidate path component: {node.attr}")
        return parent[node.attr]

    if isinstance(node, ast.Name):
        lowered = node.id.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"null", "none"}:
            return None
        if node.id not in candidate:
            raise ConstraintError(f"unknown Candidate path root: {node.id}")
        return candidate[node.id]

    if isinstance(node, ast.Constant):
        if isinstance(node.value, str | int | float | bool) or node.value is None:
            return node.value
        raise ConstraintError("unsupported literal in constraint")

    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return tuple(_evaluate_node(element, candidate) for element in node.elts)

    raise ConstraintError(
        f"unsupported constraint syntax: {node.__class__.__name__}"
    )


def _as_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ConstraintError("boolean expression operand did not evaluate to bool")
    return value


def _compare(left: object, operator: ast.cmpop, right: object) -> bool:
    if isinstance(operator, ast.Eq):
        return left == right
    if isinstance(operator, ast.NotEq):
        return left != right
    if isinstance(operator, ast.In | ast.NotIn):
        if not isinstance(right, (tuple, list, set, frozenset, str)):
            raise ConstraintError("right side of membership comparison is not a collection")
        contains = left in right
        return not contains if isinstance(operator, ast.NotIn) else contains

    if isinstance(left, bool) or isinstance(right, bool):
        raise ConstraintError("ordered comparison does not support booleans")

    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _compare_ordered(left, operator, right)
    if isinstance(left, str) and isinstance(right, str):
        return _compare_ordered(left, operator, right)

    raise ConstraintError("ordered comparison requires compatible scalar values")


def _compare_ordered(
    left: int | float | str,
    operator: ast.cmpop,
    right: int | float | str,
) -> bool:
    if isinstance(left, str) != isinstance(right, str):
        raise ConstraintError("ordered comparison requires compatible scalar values")

    if isinstance(operator, ast.Lt):
        return left < right  # type: ignore[operator]
    if isinstance(operator, ast.LtE):
        return left <= right  # type: ignore[operator]
    if isinstance(operator, ast.Gt):
        return left > right  # type: ignore[operator]
    if isinstance(operator, ast.GtE):
        return left >= right  # type: ignore[operator]
    raise ConstraintError("unsupported comparison operator")
