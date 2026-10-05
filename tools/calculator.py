"""Bounded arithmetic and subscription-cost tools."""

import ast
import math
import operator

from langchain_core.tools import tool

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_expression(expression: str) -> int | float:
    if len(expression) > 256:
        raise ValueError("수식은 256자 이하여야 합니다.")
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("수식이 너무 복잡합니다.")

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            value = _BINARY_OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            value = _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        else:
            raise ValueError("숫자와 +, -, *, /, 괄호만 사용할 수 있습니다.")
        if not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError("숫자 또는 계산 결과가 허용 범위를 벗어났습니다.")
        return value

    return evaluate(tree.body)


@tool
def calculate_math(expression: str) -> str:
    """숫자와 +, -, *, /, 괄호로 구성된 간단한 수식을 계산합니다."""
    try:
        return f"{expression} = {_evaluate_expression(expression)}"
    except (ValueError, SyntaxError, ArithmeticError, RecursionError) as error:
        return f"계산 오류: {error}"


@tool
def calculate_subscription_cost(tool_names: list[str], tool_prices: list[float]) -> str:
    """선택된 AI 도구들의 월간 구독료(USD)를 합산합니다."""
    if len(tool_names) != len(tool_prices):
        return "오류: 도구 이름과 가격 리스트의 길이가 일치하지 않습니다."
    if any(not math.isfinite(price) or price < 0 for price in tool_prices):
        return "오류: 가격은 0 이상의 유한한 숫자여야 합니다."
    total = sum(tool_prices)
    if not math.isfinite(total * 12):
        return "오류: 합계가 계산 범위를 벗어났습니다."
    lines = ["## 월간 구독료 계산 결과\n"]
    for name, price in zip(tool_names, tool_prices, strict=True):
        lines.append(f"- **{name}**: 무료" if price == 0 else f"- **{name}**: {price:.2f} USD/월")
    lines.extend(
        [
            f"\n- **월간 합계**: {total:.2f} USD",
            f"- **연간 합계**: {total * 12:.2f} USD",
        ]
    )
    if total > 50:
        lines.append("\n월 50 USD 이상의 비용이 예상됩니다. 무료 대안도 확인해보세요.")
    return "\n".join(lines)


def calculate_tools_cost(tools: list[dict]) -> dict:
    """Catalog prices may be stale; expose the source fields instead of inventing numbers."""
    return {
        "breakdown": [
            {
                "name": item.get("name", "Unknown"),
                "pricing_model": item.get("pricing_model", ""),
                "pricing_notes": item.get("pricing_notes", ""),
            }
            for item in tools
        ],
        "note": "정확한 비용은 각 서비스 공식 사이트에서 확인하세요.",
    }
