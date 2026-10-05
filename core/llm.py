"""
Core LLM - LLM 팩토리 함수
"""

from langchain_openai import ChatOpenAI

from core.config import LLM_MODEL


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    LLM 인스턴스 반환

    Args:
        temperature: 생성 온도 (0.0 ~ 1.0)

    Returns:
        ChatOpenAI 인스턴스
    """
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature, timeout=30, max_retries=2)
