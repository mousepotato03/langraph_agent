"""Shared application service for API and UI. Sessions are local to one process."""

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from time import monotonic
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.types import Command

from agent.graph import create_agent_graph, create_initial_state
from agent.hitl import analyze_user_intent
from core.config import GRAPH_RECURSION_LIMIT, MAX_ACTIVE_SESSIONS, SESSION_TTL_SECONDS
from core.llm import get_llm


class ServiceError(Exception):
    status_code = 500


class SessionNotFound(ServiceError):
    status_code = 404


class SessionConflict(ServiceError):
    status_code = 409


class SessionCapacityExceeded(ServiceError):
    status_code = 503


class AgentExecutionError(ServiceError):
    status_code = 502


@dataclass(frozen=True)
class ChatResult:
    thread_id: str
    status: Literal["pending_approval", "completed", "cancelled"]
    message: str
    is_complex: bool
    plan: list[dict[str, str]] | None = None
    final_guide: str | None = None


@dataclass
class Session:
    graph: Any
    user_id: str
    updated_at: float
    busy: bool = True
    config: dict = field(default_factory=dict)


class AgentService:
    def __init__(
        self,
        *,
        graph_factory: Callable = create_agent_graph,
        llm_factory: Callable = get_llm,
        ttl_seconds: float = SESSION_TTL_SECONDS,
        max_sessions: int = MAX_ACTIVE_SESSIONS,
        clock: Callable[[], float] = monotonic,
    ):
        if ttl_seconds <= 0 or max_sessions < 1:
            raise ValueError("Session limits must be positive")
        self._graph_factory = graph_factory
        self._llm_factory = llm_factory
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._clock = clock
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()

    def _purge_expired(self):
        now = self._clock()
        expired = [
            key
            for key, session in self._sessions.items()
            if not session.busy and now - session.updated_at >= self._ttl
        ]
        for key in expired:
            del self._sessions[key]

    def _get_session(self, thread_id: str, user_id: str) -> Session:
        self._purge_expired()
        session = self._sessions.get(thread_id)
        if session is None or session.user_id != user_id:
            raise SessionNotFound("세션을 찾을 수 없거나 만료되었습니다.")
        if session.busy:
            raise SessionConflict("이 세션의 요청이 이미 실행 중입니다.")
        return session

    def has_pending(self, thread_id: str, user_id: str) -> bool:
        with self._lock:
            self._purge_expired()
            session = self._sessions.get(thread_id)
            return session is not None and session.user_id == user_id

    def start(
        self,
        query: str,
        user_id: str = "default_user",
        *,
        thread_id: str | None = None,
        chat_history: list[BaseMessage] | None = None,
    ) -> ChatResult:
        if not query.strip() or not user_id.strip():
            raise ValueError("질문과 사용자 ID를 입력해주세요.")
        thread_id = thread_id or str(uuid4())
        with self._lock:
            self._purge_expired()
            if thread_id in self._sessions:
                raise SessionConflict("이미 사용 중인 thread_id입니다.")
            if len(self._sessions) >= self._max_sessions:
                raise SessionCapacityExceeded(
                    "대기 중인 세션이 많습니다. 잠시 후 다시 시도해주세요."
                )
            session = Session(
                graph=self._graph_factory(),
                user_id=user_id,
                updated_at=self._clock(),
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": GRAPH_RECURSION_LIMIT,
                },
            )
            self._sessions[thread_id] = session
        return self._run(
            thread_id, session, create_initial_state(query.strip(), user_id, chat_history)
        )

    def resume(
        self, thread_id: str, user_id: str, action: str, feedback: str | None = None
    ) -> ChatResult:
        if action not in ("approve", "modify", "cancel", "clarify"):
            raise ValueError("알 수 없는 승인 동작입니다.")
        if action == "modify" and (not feedback or not feedback.strip()):
            raise ValueError("계획 수정 내용을 입력해주세요.")
        with self._lock:
            session = self._get_session(thread_id, user_id)
            session.busy = True
        return self._run(
            thread_id, session, Command(resume={"action": action, "feedback": feedback})
        )

    def respond(self, thread_id: str, user_id: str, text: str) -> ChatResult:
        """Only UI natural-language responses need intent classification."""
        with self._lock:
            session = self._get_session(thread_id, user_id)
            session.busy = True
        try:
            state = session.graph.get_state(session.config).values
            decision = analyze_user_intent(
                text,
                "\n".join(state["sub_tasks"]),
                llm_factory=self._llm_factory,
            )
        except Exception:
            with self._lock:
                session.busy = False
            raise
        return self._run(thread_id, session, Command(resume=decision))

    def _run(self, thread_id: str, session: Session, graph_input) -> ChatResult:
        try:
            session.graph.invoke(graph_input, config=session.config)
            snapshot = session.graph.get_state(session.config)
            state = snapshot.values
            if state["approval_status"] == "cancelled":
                result = ChatResult(thread_id, "cancelled", "작업이 취소되었습니다.", True)
            elif snapshot.interrupts:
                if state["approval_status"] != "pending":
                    raise RuntimeError("Unexpected interrupt outside plan review")
                plan = [
                    {"id": f"task_{i}", "description": task}
                    for i, task in enumerate(state["sub_tasks"], 1)
                ]
                summary = "\n".join(f"{item['id']}: {item['description']}" for item in plan)
                result = ChatResult(
                    thread_id,
                    "pending_approval",
                    f"{state['approval_message']}\n\n분석: {state['plan_analysis']}\n\n{summary}",
                    True,
                    plan=plan,
                )
            else:
                answer = state["final_guide"] or state["final_answer"]
                if not answer or snapshot.next:
                    raise RuntimeError("Graph ended without a final answer")
                result = ChatResult(
                    thread_id,
                    "completed",
                    "답변이 완료되었습니다.",
                    state["is_complex_task"],
                    final_guide=answer,
                )
        except Exception as error:
            with self._lock:
                self._sessions.pop(thread_id, None)
            raise AgentExecutionError(
                "응답 생성에 실패했습니다. 새 대화로 다시 시도해주세요."
            ) from error
        with self._lock:
            if result.status == "pending_approval":
                session.busy = False
                session.updated_at = self._clock()
            else:
                self._sessions.pop(thread_id, None)
        return result

    def discard(self, thread_id: str | None, user_id: str):
        """UI reset releases an unfinished graph without executing any more nodes."""
        if thread_id is None:
            return
        with self._lock:
            self._purge_expired()
            if thread_id not in self._sessions:
                return
            self._get_session(thread_id, user_id)
            del self._sessions[thread_id]

    def close(self):
        with self._lock:
            self._sessions.clear()
