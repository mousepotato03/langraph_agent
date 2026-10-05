"""Generate a guide from the completed recommendations and their evidence."""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agent.dependencies import AgentDependencies
from agent.state import AgentState
from prompts.formatters import format_user_profile
from prompts.guide import GUIDE_GENERATION_SYSTEM_PROMPT, GUIDE_GENERATION_USER_TEMPLATE


def guide_generation_node(state: AgentState, *, deps: AgentDependencies) -> dict:
    prompt = GUIDE_GENERATION_USER_TEMPLATE.format(
        user_query=state["user_query"],
        sub_tasks="\n".join(f"{i}. {task}" for i, task in enumerate(state["sub_tasks"], 1)),
        tool_recommendations="\n".join(
            f"### {task_id}\n{recommendation}"
            for task_id, recommendation in state["tool_recommendations"].items()
        ),
        retrieved_docs=json.dumps(state["retrieved_docs"], ensure_ascii=False),
        user_profile=format_user_profile(state["user_profile"]),
    )
    response = deps.llm_factory(temperature=0.7).invoke(
        [
            SystemMessage(content=GUIDE_GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return {"final_guide": response.content, "messages": [response]}
