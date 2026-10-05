"""Simple-question tool execution uses the same registry as model binding."""

from agent.dependencies import AgentDependencies
from agent.state import AgentState
from agent.tool_execution import execute_pending_tools


def simple_tool_executor(state: AgentState, *, deps: AgentDependencies) -> dict:
    return execute_pending_tools(state, deps=deps, simple=True)
