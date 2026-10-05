"""Recommendation tool execution."""

from agent.dependencies import AgentDependencies
from agent.state import AgentState
from agent.tool_execution import execute_pending_tools


def tool_executor_node(state: AgentState, *, deps: AgentDependencies) -> dict:
    return execute_pending_tools(state, deps=deps, simple=False)
