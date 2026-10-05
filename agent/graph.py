"""Build the graph independently of API/UI session handling."""

from functools import partial

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from agent.dependencies import AgentDependencies
from agent.nodes import (
    guide_generation_node,
    human_approval_node,
    llm_router_node,
    planning_node,
    recommend_tool_node,
    reflection_node,
    simple_llm_node,
    simple_tool_executor,
    tool_executor_node,
)
from agent.routing import (
    route_after_approval,
    route_after_llm_router,
    route_after_recommend,
    route_after_simple_llm,
)
from agent.state import AgentState


def create_initial_state(
    user_query: str,
    user_id: str = "default_user",
    chat_history: list[BaseMessage] | None = None,
) -> AgentState:
    return {
        "messages": [*(chat_history or []), HumanMessage(content=user_query)],
        "user_id": user_id,
        "user_query": user_query,
        "user_profile": None,
        "is_complex_task": False,
        "sub_tasks": [],
        "plan_analysis": "",
        "approval_status": "pending",
        "approval_message": "계획을 검토하고 승인·수정·취소를 선택해주세요.",
        "user_feedback": None,
        "current_task_idx": 0,
        "tool_recommendations": {},
        "pending_tool_calls": [],
        "tool_call_count": 0,
        "simple_tool_count": 0,
        "task_messages": [],
        "task_documents": [],
        "task_recommendation": None,
        "retrieved_docs": [],
        "final_guide": None,
        "final_answer": None,
    }


def create_agent_graph(dependencies: AgentDependencies | None = None, *, checkpointer=None):
    dependencies = dependencies or AgentDependencies()
    workflow = StateGraph(AgentState)
    nodes = {
        "llm_router": llm_router_node,
        "planning_node": planning_node,
        "human_approval_node": human_approval_node,
        "recommend_tool_node": recommend_tool_node,
        "tool_executor": tool_executor_node,
        "guide_generation_node": guide_generation_node,
        "reflection_node": reflection_node,
        "simple_llm_node": simple_llm_node,
        "simple_executor": simple_tool_executor,
    }
    for name, node in nodes.items():
        workflow.add_node(name, partial(node, deps=dependencies))

    workflow.set_entry_point("llm_router")
    workflow.add_conditional_edges(
        "llm_router", route_after_llm_router, ["planning_node", "simple_llm_node"]
    )
    workflow.add_conditional_edges(
        "simple_llm_node", route_after_simple_llm, ["simple_executor", "reflection_node"]
    )
    workflow.add_edge("simple_executor", "simple_llm_node")
    workflow.add_edge("planning_node", "human_approval_node")
    workflow.add_conditional_edges(
        "human_approval_node",
        route_after_approval,
        ["recommend_tool_node", "human_approval_node", END],
    )
    workflow.add_conditional_edges(
        "recommend_tool_node",
        route_after_recommend,
        ["tool_executor", "recommend_tool_node", "guide_generation_node"],
    )
    workflow.add_edge("tool_executor", "recommend_tool_node")
    workflow.add_edge("guide_generation_node", "reflection_node")
    workflow.add_edge("reflection_node", END)
    return workflow.compile(checkpointer=checkpointer or InMemorySaver())
