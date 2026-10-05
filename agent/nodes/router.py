"""Classify a request without appending internal classifier messages to the chat."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.dependencies import AgentDependencies
from agent.state import AgentState
from core.utils import extract_json
from prompts.formatters import format_user_profile
from prompts.router import LLM_ROUTER_SYSTEM_PROMPT, LLM_ROUTER_USER_TEMPLATE

logger = logging.getLogger(__name__)


def llm_router_node(state: AgentState, *, deps: AgentDependencies) -> dict:
    profile = deps.memory_factory().load_user_profile(state["user_id"])
    response = deps.llm_factory(temperature=0.1).invoke(
        [
            SystemMessage(content=LLM_ROUTER_SYSTEM_PROMPT),
            HumanMessage(
                content=LLM_ROUTER_USER_TEMPLATE.format(
                    user_query=state["user_query"], user_profile=format_user_profile(profile)
                )
            ),
        ]
    )
    try:
        result = json.loads(extract_json(response.content))
        is_complex = result["is_complex"]
        if not isinstance(is_complex, bool):
            raise ValueError("is_complex must be a boolean")
    except (ValueError, KeyError, TypeError):
        logger.warning("Invalid classification; requesting a plan review")
        is_complex = True
    return {"is_complex_task": is_complex, "user_profile": profile}
