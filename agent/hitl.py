"""Natural-language interpretation is separate from explicit approval decisions."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm import get_llm
from core.utils import extract_json, validate_subtasks
from prompts.intent import (
    INTENT_ANALYSIS_SYSTEM_PROMPT,
    INTENT_ANALYSIS_USER_TEMPLATE,
    MODIFY_PLAN_SYSTEM_PROMPT,
    MODIFY_PLAN_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


def analyze_user_intent(user_text: str, plan_summary: str, *, llm_factory=get_llm) -> dict:
    """Ambiguous or failed classification must keep the approval gate closed."""
    try:
        response = llm_factory(temperature=0.1).invoke(
            [
                SystemMessage(content=INTENT_ANALYSIS_SYSTEM_PROMPT),
                HumanMessage(
                    content=INTENT_ANALYSIS_USER_TEMPLATE.format(
                        plan_summary=plan_summary, user_response=user_text
                    )
                ),
            ]
        )
        result = json.loads(extract_json(response.content))
        action = result.get("intent")
        if action not in ("approve", "modify", "cancel"):
            return {"action": "clarify", "feedback": ""}
        feedback = result.get("feedback", "")
        if action == "modify" and (not isinstance(feedback, str) or not feedback.strip()):
            return {"action": "clarify", "feedback": ""}
        return {"action": action, "feedback": feedback}
    except Exception:
        logger.warning("Could not classify approval response", exc_info=True)
        return {"action": "clarify", "feedback": ""}


def modify_subtasks(
    current_subtasks: list[str], feedback: str, *, llm_factory=get_llm
) -> list[str]:
    """Raise on an invalid revision rather than approving the old plan."""
    response = llm_factory(temperature=0.3).invoke(
        [
            SystemMessage(content=MODIFY_PLAN_SYSTEM_PROMPT),
            HumanMessage(
                content=MODIFY_PLAN_USER_TEMPLATE.format(
                    current_plan="\n".join(f"- {task}" for task in current_subtasks),
                    feedback=feedback,
                )
            ),
        ]
    )
    return validate_subtasks(json.loads(extract_json(response.content)))
