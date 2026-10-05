"""Validate the generated plan before exposing it for review."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.dependencies import AgentDependencies
from agent.state import AgentState
from core.utils import extract_json, validate_subtasks
from prompts.formatters import format_user_profile
from prompts.planning import PLAN_SYSTEM_PROMPT, PLAN_USER_TEMPLATE

logger = logging.getLogger(__name__)


def planning_node(state: AgentState, *, deps: AgentDependencies) -> dict:
    response = deps.llm_factory(temperature=0.5).invoke(
        [
            SystemMessage(content=PLAN_SYSTEM_PROMPT),
            HumanMessage(
                content=PLAN_USER_TEMPLATE.format(
                    user_query=state["user_query"],
                    user_profile=format_user_profile(state["user_profile"]),
                )
            ),
        ]
    )
    try:
        plan = json.loads(extract_json(response.content))
        tasks = validate_subtasks(plan["subtasks"])
        analysis = plan.get("analysis", "")
        if not isinstance(analysis, str):
            raise ValueError("analysis must be a string")
    except (ValueError, KeyError, TypeError):
        logger.warning("Invalid plan; using the original request as one reviewable task")
        tasks = [state["user_query"]]
        analysis = "세부 계획 생성에 실패하여 원래 요청을 하나의 작업으로 제시합니다."
    return {
        "plan_analysis": analysis,
        "sub_tasks": tasks,
        "approval_status": "pending",
        "approval_message": "계획을 검토하고 승인·수정·취소를 선택해주세요.",
    }
