"""Simple ReAct loop preserves the original AI/Tool message order."""

from langchain_core.messages import HumanMessage, SystemMessage

from agent.dependencies import AgentDependencies
from agent.state import AgentState
from core.config import MAX_TOOL_CALLS_PER_TASK
from prompts.formatters import format_user_profile
from prompts.simple import SIMPLE_REACT_SYSTEM_PROMPT
from tools.registry import get_simple_tools


def simple_llm_node(state: AgentState, *, deps: AgentDependencies) -> dict:
    llm = deps.llm_factory(temperature=0.5)
    messages = [
        SystemMessage(
            content=(
                SIMPLE_REACT_SYSTEM_PROMPT
                + "\n\n사용자 프로필:\n"
                + format_user_profile(state["user_profile"])
            )
        ),
        *state["messages"],
    ]
    if state["simple_tool_count"] >= MAX_TOOL_CALLS_PER_TASK:
        messages.append(
            HumanMessage(
                content=(
                    "도구 호출 한도에 도달했습니다. 지금까지의 결과로 답변하고 "
                    "확인하지 못한 정보는 명확히 밝혀주세요."
                )
            )
        )
        response = llm.invoke(messages)
    else:
        response = llm.bind_tools(get_simple_tools(), parallel_tool_calls=False).invoke(messages)

    if response.tool_calls:
        return {
            "pending_tool_calls": response.tool_calls,
            "messages": [response],
        }
    return {
        "pending_tool_calls": [],
        "final_answer": response.content,
        "messages": [response],
    }
