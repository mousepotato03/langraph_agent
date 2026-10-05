import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent.dependencies import AgentDependencies
from agent.graph import create_initial_state
from agent.hitl import analyze_user_intent
from agent.tool_execution import execute_pending_tools
from core.utils import extract_json
from tests.fakes import ScriptedModel
from tools.calculator import calculate_math, calculate_subscription_cost
from tools.registry import get_simple_tools


def test_simple_registry_includes_the_advertised_math_tool():
    assert "calculate_math" in {tool.name for tool in get_simple_tools()}


def test_batch_returns_every_original_call_id_even_when_execution_fails():
    state = create_initial_state("계산", "alice")
    state["pending_tool_calls"] = [
        {
            "name": "calculate_math",
            "args": {"expression": "1+2"},
            "id": "good",
            "type": "tool_call",
        },
        {
            "name": "google_search_tool",
            "args": {"query": "video"},
            "id": "failed",
            "type": "tool_call",
        },
        {"name": "not_a_tool", "args": {}, "id": "unknown", "type": "tool_call"},
        {"name": "get_current_time", "args": {}, "id": "over-budget", "type": "tool_call"},
    ]
    invoked = []

    def runner(name, args):
        invoked.append(name)
        if name == "google_search_tool":
            raise RuntimeError("private upstream details")
        return "3"

    result = execute_pending_tools(state, deps=AgentDependencies(tool_runner=runner), simple=True)
    assert all(isinstance(item, ToolMessage) for item in result["messages"])
    assert [item.tool_call_id for item in result["messages"]] == [
        "good",
        "failed",
        "unknown",
        "over-budget",
    ]
    assert [item.status for item in result["messages"]] == ["success", "error", "error", "error"]
    assert invoked == ["calculate_math", "google_search_tool"]
    assert result["simple_tool_count"] == 3
    assert "private upstream details" not in str(result)
    assert not result["pending_tool_calls"]


def test_memory_tools_use_the_current_user_even_if_model_supplies_another_id():
    state = create_initial_state("선호도 저장", "alice")
    state["pending_tool_calls"] = [
        {"name": "read_memory", "args": {"user_id": "bob"}, "id": "read", "type": "tool_call"}
    ]
    arguments = []
    result = execute_pending_tools(
        state,
        deps=AgentDependencies(tool_runner=lambda name, args: arguments.append(args) or "{}"),
        simple=False,
    )
    assert arguments == [{"user_id": "alice"}]
    assert result["tool_call_count"] == 1


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("22 * 34", "748"),
        ("(10 + 2) / 4", "3.0"),
        ("-3 * -2", "6"),
    ],
)
def test_supported_arithmetic(expression, expected):
    assert calculate_math.invoke({"expression": expression}) == f"{expression} = {expected}"


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "9 ** 9 ** 9",
        "1 / 0",
        "1e999",
        "1+" * 200,
        "[" + "0," * 30 + "0]",
        "True + 1",
    ],
)
def test_invalid_or_unbounded_arithmetic_returns_an_error(expression):
    assert calculate_math.invoke({"expression": expression}).startswith("계산 오류:")


@pytest.mark.parametrize("prices", [[-1], [float("inf")], [float("nan")]])
def test_subscription_prices_must_be_nonnegative_finite_numbers(prices):
    assert calculate_subscription_cost.invoke(
        {"tool_names": ["Example"], "tool_prices": prices}
    ).startswith("오류:")


@pytest.mark.parametrize(
    "response", ["not json", '{"intent": "surprise"}', "[]", '{"intent": "modify", "feedback": ""}']
)
def test_failed_or_ambiguous_intent_is_never_approval(response):
    assert (
        analyze_user_intent(
            "음...", "작업", llm_factory=ScriptedModel([AIMessage(content=response)])
        )["action"]
        == "clarify"
    )


def test_intent_model_outage_keeps_gate_closed():
    assert (
        analyze_user_intent("음...", "작업", llm_factory=ScriptedModel([RuntimeError("offline")]))[
            "action"
        ]
        == "clarify"
    )


def test_json_extraction_handles_fences_and_braces_inside_strings():
    fence = chr(96) * 3
    text = "설명\n" + fence + 'json\n{"notes": "a { brace } and \\"quote\\""}\n' + fence
    assert json.loads(extract_json(text))["notes"] == 'a { brace } and "quote"'
