"""Recommend per task, keeping evidence and ReAct messages isolated between tasks."""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agent.dependencies import AgentDependencies
from agent.state import AgentState
from core.config import MAX_TOOL_CALLS_PER_TASK
from prompts.formatters import format_user_profile
from prompts.recommend import RECOMMEND_TOOL_SYSTEM_PROMPT, RECOMMEND_TOOL_USER_TEMPLATE
from tools.registry import get_all_tools


def finish_task(state: AgentState, recommendation: str) -> dict:
    recommendations = dict(state["tool_recommendations"])
    recommendations[f"task_{state['current_task_idx'] + 1}"] = recommendation
    return {
        "tool_recommendations": recommendations,
        "current_task_idx": state["current_task_idx"] + 1,
        "tool_call_count": 0,
        "pending_tool_calls": [],
        "task_messages": [],
        "task_documents": [],
        "task_recommendation": None,
    }


def recommend_tool_node(state: AgentState, *, deps: AgentDependencies) -> dict:
    if state["current_task_idx"] >= len(state["sub_tasks"]):
        return {"pending_tool_calls": []}
    recommended = state["task_recommendation"]
    if recommended:
        score = recommended.get("scores", {}).get("final_score", 0)
        result = (
            f"**{recommended.get('name', '이름 없음')}** (검색 점수: {score:.2f})\n"
            f"{recommended.get('description', '')}"
        )
        return finish_task(state, result)

    llm = deps.llm_factory(temperature=0.3)
    prompt = RECOMMEND_TOOL_USER_TEMPLATE.format(
        current_task=state["sub_tasks"][state["current_task_idx"]],
        previous_results=json.dumps(state["task_documents"], ensure_ascii=False),
        user_profile=format_user_profile(state["user_profile"]),
    )
    messages = [
        SystemMessage(content=RECOMMEND_TOOL_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
        *state["task_messages"],
    ]
    if state["tool_call_count"] >= MAX_TOOL_CALLS_PER_TASK:
        messages.append(
            HumanMessage(
                content=(
                    "도구 호출 한도에 도달했습니다. 수집한 근거로 추천을 마무리하고, "
                    "적합한 도구를 찾지 못했으면 추가 확인이 필요하다고 밝혀주세요."
                )
            )
        )
        response = llm.invoke(messages)
    else:
        response = llm.bind_tools(get_all_tools(), parallel_tool_calls=False).invoke(messages)
    if response.tool_calls:
        return {
            "pending_tool_calls": response.tool_calls,
            "task_messages": [*state["task_messages"], response],
            "messages": [response],
        }
    return {**finish_task(state, response.content), "messages": [response]}
