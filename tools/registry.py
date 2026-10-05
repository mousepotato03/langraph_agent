"""One registry is used both for model binding and execution."""

from langchain_core.tools import BaseTool

from tools.calculator import calculate_math, calculate_subscription_cost
from tools.memory_tools import read_memory, write_memory
from tools.search import google_search_tool, retrieve_docs
from tools.time_tools import check_tool_freshness, get_current_time

TOOLS: dict[str, BaseTool] = {
    item.name: item
    for item in (
        retrieve_docs,
        read_memory,
        write_memory,
        google_search_tool,
        calculate_subscription_cost,
        calculate_math,
        check_tool_freshness,
        get_current_time,
    )
}
SIMPLE_TOOL_NAMES = frozenset(
    {
        "get_current_time",
        "google_search_tool",
        "calculate_math",
        "calculate_subscription_cost",
        "check_tool_freshness",
    }
)


def get_all_tools() -> list[BaseTool]:
    return list(TOOLS.values())


def get_simple_tools() -> list[BaseTool]:
    return [item for name, item in TOOLS.items() if name in SIMPLE_TOOL_NAMES]


def execute_tool(tool_name: str, args: dict):
    if tool_name not in TOOLS:
        raise ValueError(f"알 수 없는 도구: {tool_name}")
    return TOOLS[tool_name].invoke(args)
