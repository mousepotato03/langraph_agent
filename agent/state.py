"""Checkpointed state shared by graph nodes."""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, ToolCall
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    user_query: str
    user_profile: dict[str, Any] | None
    is_complex_task: bool
    sub_tasks: list[str]
    plan_analysis: str
    approval_status: Literal["pending", "approved", "cancelled"]
    approval_message: str
    user_feedback: str | None
    current_task_idx: int
    tool_recommendations: dict[str, str]
    # A batch retains every call ID, including calls that fail or exceed the budget.
    pending_tool_calls: list[ToolCall]
    tool_call_count: int
    simple_tool_count: int
    # Replaced/reset between subtasks; retrieved_docs retains evidence for the guide.
    task_messages: list[BaseMessage]
    task_documents: list[dict[str, Any]]
    task_recommendation: dict[str, Any] | None
    retrieved_docs: list[dict[str, Any]]
    final_guide: str | None
    final_answer: str | None
