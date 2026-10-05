"""Execute complete call batches and always return matching ToolMessage IDs."""

import json
import logging

from langchain_core.messages import ToolMessage

from agent.dependencies import AgentDependencies
from agent.state import AgentState
from core.config import MAX_TOOL_CALLS_PER_TASK
from tools.registry import SIMPLE_TOOL_NAMES, TOOLS

logger = logging.getLogger(__name__)


def execute_pending_tools(state: AgentState, *, deps: AgentDependencies, simple: bool) -> dict:
    counter_key = "simple_tool_count" if simple else "tool_call_count"
    count = state[counter_key]
    allowed = SIMPLE_TOOL_NAMES if simple else TOOLS.keys()
    messages = []
    documents = []
    recommended = None
    for call in state["pending_tool_calls"]:
        name = call["name"]
        args = dict(call["args"])
        # Profiles are scoped by the trusted graph state, never by generated arguments.
        if name in ("read_memory", "write_memory"):
            args["user_id"] = state["user_id"]
        try:
            if count >= MAX_TOOL_CALLS_PER_TASK:
                raise ValueError("도구 호출 한도에 도달했습니다.")
            count += 1
            if name not in allowed:
                raise ValueError(f"허용되지 않은 도구: {name}")
            result = deps.tool_runner(name, args)
            observation = (
                result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            )
            messages.append(ToolMessage(content=observation, tool_call_id=call["id"], name=name))
        except Exception:
            logger.warning("Tool failed: %s", name, exc_info=True)
            messages.append(
                ToolMessage(
                    content=f"도구 {name} 실행에 실패했습니다. 입력과 설정을 확인해주세요.",
                    tool_call_id=call["id"],
                    name=name,
                    status="error",
                )
            )
            continue

        if name in ("retrieve_docs", "google_search_tool"):
            try:
                data = json.loads(observation)
                if name == "retrieve_docs":
                    candidate = data.get("recommended_tool")
                    if isinstance(candidate, dict):
                        documents.append(candidate)
                        if data.get("should_fallback") is False:
                            recommended = candidate
                else:
                    results = data.get("results", [])
                    if isinstance(results, list):
                        documents.extend(item for item in results if isinstance(item, dict))
            except (ValueError, AttributeError):
                logger.warning("Invalid search result from %s", name)

    updates = {
        "pending_tool_calls": [],
        counter_key: count,
        "messages": messages,
        "retrieved_docs": [*state["retrieved_docs"], *documents],
    }
    if not simple:
        updates.update(
            {
                "task_messages": [*state["task_messages"], *messages],
                "task_documents": [*state["task_documents"], *documents],
                "task_recommendation": recommended,
            }
        )
    return updates
