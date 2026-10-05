"""Dependencies can be replaced without API keys, model downloads, or network calls."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.llm import get_llm
from core.memory import get_memory_manager
from tools.registry import execute_tool


@dataclass(frozen=True)
class AgentDependencies:
    llm_factory: Callable[..., Any] = get_llm
    memory_factory: Callable[[], Any] = get_memory_manager
    tool_runner: Callable[[str, dict], Any] = execute_tool
