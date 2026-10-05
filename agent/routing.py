"""Graph routing contains no model or tool calls."""

from langgraph.graph import END

from agent.state import AgentState


def route_after_llm_router(state: AgentState) -> str:
    return "planning_node" if state["is_complex_task"] else "simple_llm_node"


def route_after_approval(state: AgentState) -> str:
    if state["approval_status"] == "cancelled":
        return END
    if state["approval_status"] == "approved":
        return "recommend_tool_node"
    return "human_approval_node"


def route_after_simple_llm(state: AgentState) -> str:
    return "simple_executor" if state["pending_tool_calls"] else "reflection_node"


def route_after_recommend(state: AgentState) -> str:
    if state["pending_tool_calls"]:
        return "tool_executor"
    if state["current_task_idx"] < len(state["sub_tasks"]):
        return "recommend_tool_node"
    return "guide_generation_node"
