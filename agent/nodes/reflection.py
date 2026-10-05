"""Extract preferences after either response branch; memory failures are non-fatal."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.dependencies import AgentDependencies
from agent.state import AgentState
from core.utils import extract_json, merge_preferences
from prompts.reflection import MEMORY_EXTRACTOR_SYSTEM_PROMPT, MEMORY_EXTRACTOR_USER_TEMPLATE

logger = logging.getLogger(__name__)


def reflection_node(state: AgentState, *, deps: AgentDependencies) -> dict:
    answer = state["final_guide"] or state["final_answer"] or ""
    try:
        memory = deps.memory_factory()
        # A write_memory call may have updated the profile during this run.
        existing_profile = memory.load_user_profile(state["user_id"]) or state["user_profile"]
        response = deps.llm_factory(temperature=0.3).invoke(
            [
                SystemMessage(content=MEMORY_EXTRACTOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=MEMORY_EXTRACTOR_USER_TEMPLATE.format(
                        conversation=f"사용자: {state['user_query']}\n\n에이전트: {answer}",
                        existing_profile=json.dumps(existing_profile, ensure_ascii=False),
                    )
                ),
            ]
        )
        preferences = merge_preferences(
            existing_profile, json.loads(extract_json(response.content))
        )
        if not memory.save_user_profile(state["user_id"], preferences):
            logger.warning("Profile storage failed")
    except Exception:
        logger.warning("Reflection failed; preserving the response", exc_info=True)
    return {}
