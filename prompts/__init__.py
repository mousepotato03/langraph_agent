"""프롬프트 템플릿 모듈"""

from prompts.formatters import (
    format_guides,
    format_plan_summary,
    format_search_results,
    format_user_profile,
)
from prompts.guide import (
    GUIDE_GENERATION_SYSTEM_PROMPT,
    GUIDE_GENERATION_USER_TEMPLATE,
    GUIDE_SIMPLE_QA_SYSTEM_PROMPT,
    GUIDE_SIMPLE_QA_USER_TEMPLATE,
)
from prompts.intent import (
    INTENT_ANALYSIS_SYSTEM_PROMPT,
    INTENT_ANALYSIS_USER_TEMPLATE,
    MODIFY_PLAN_SYSTEM_PROMPT,
    MODIFY_PLAN_USER_TEMPLATE,
)
from prompts.planning import PLAN_SYSTEM_PROMPT, PLAN_USER_TEMPLATE
from prompts.recommend import RECOMMEND_TOOL_SYSTEM_PROMPT, RECOMMEND_TOOL_USER_TEMPLATE
from prompts.reflection import MEMORY_EXTRACTOR_SYSTEM_PROMPT, MEMORY_EXTRACTOR_USER_TEMPLATE
from prompts.router import LLM_ROUTER_SYSTEM_PROMPT, LLM_ROUTER_USER_TEMPLATE

# 레거시 호환용 (기존 코드에서 사용하는 이름들)
SYNTHESIZE_SYSTEM_PROMPT = GUIDE_GENERATION_SYSTEM_PROMPT
SYNTHESIZE_USER_TEMPLATE = GUIDE_GENERATION_USER_TEMPLATE
HUMAN_REVIEW_MESSAGE = "이 계획대로 진행할까요? (승인/수정/취소)"

__all__ = [
    # Router
    "LLM_ROUTER_SYSTEM_PROMPT",
    "LLM_ROUTER_USER_TEMPLATE",
    # Planning
    "PLAN_SYSTEM_PROMPT",
    "PLAN_USER_TEMPLATE",
    # Recommend
    "RECOMMEND_TOOL_SYSTEM_PROMPT",
    "RECOMMEND_TOOL_USER_TEMPLATE",
    # Guide
    "GUIDE_GENERATION_SYSTEM_PROMPT",
    "GUIDE_GENERATION_USER_TEMPLATE",
    "GUIDE_SIMPLE_QA_SYSTEM_PROMPT",
    "GUIDE_SIMPLE_QA_USER_TEMPLATE",
    # Reflection
    "MEMORY_EXTRACTOR_SYSTEM_PROMPT",
    "MEMORY_EXTRACTOR_USER_TEMPLATE",
    # Intent
    "INTENT_ANALYSIS_SYSTEM_PROMPT",
    "INTENT_ANALYSIS_USER_TEMPLATE",
    "MODIFY_PLAN_SYSTEM_PROMPT",
    "MODIFY_PLAN_USER_TEMPLATE",
    # Formatters
    "format_search_results",
    "format_plan_summary",
    "format_user_profile",
    "format_guides",
    # Legacy
    "SYNTHESIZE_SYSTEM_PROMPT",
    "SYNTHESIZE_USER_TEMPLATE",
    "HUMAN_REVIEW_MESSAGE",
]
