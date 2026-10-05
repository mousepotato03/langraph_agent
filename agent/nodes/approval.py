"""The graph itself owns the approval gate, including repeated plan revisions."""

import logging

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from agent.dependencies import AgentDependencies
from agent.hitl import modify_subtasks
from agent.state import AgentState

logger = logging.getLogger(__name__)


def human_approval_node(state: AgentState, *, deps: AgentDependencies) -> dict:
    # Keep all effects AFTER interrupt(): this node restarts when resumed.
    decision = interrupt(
        {
            "sub_tasks": state["sub_tasks"],
            "analysis": state["plan_analysis"],
            "message": state["approval_message"],
        }
    )
    action = decision.get("action") if isinstance(decision, dict) else None
    if action == "approve":
        return {
            "approval_status": "approved",
            "messages": [AIMessage(content="계획을 승인했습니다. 추천을 시작합니다.")],
        }
    if action == "cancel":
        return {
            "approval_status": "cancelled",
            "final_guide": "작업이 취소되었습니다.",
        }
    if action == "modify":
        feedback = decision.get("feedback")
        if isinstance(feedback, str) and feedback.strip():
            try:
                tasks = modify_subtasks(state["sub_tasks"], feedback, llm_factory=deps.llm_factory)
            except Exception:
                logger.warning("Could not revise plan", exc_info=True)
                return {
                    "approval_status": "pending",
                    "approval_message": "계획을 수정하지 못했습니다. 수정 내용을 다시 알려주세요.",
                }
            return {
                "sub_tasks": tasks,
                "user_feedback": feedback,
                "approval_status": "pending",
                "approval_message": "계획을 수정했습니다. 변경된 계획을 다시 승인해주세요.",
                "current_task_idx": 0,
                "tool_recommendations": {},
                "pending_tool_calls": [],
                "tool_call_count": 0,
                "task_messages": [],
                "task_documents": [],
                "task_recommendation": None,
                "retrieved_docs": [],
            }
    return {
        "approval_status": "pending",
        "approval_message": "승인·수정·취소 중 원하는 작업을 명확히 알려주세요.",
    }
