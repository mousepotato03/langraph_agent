from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Event

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agent.dependencies import AgentDependencies
from agent.graph import create_agent_graph
from app.api.routes import create_app
from app.service import (
    AgentService,
    SessionCapacityExceeded,
    SessionConflict,
    SessionNotFound,
)
from app.ui.gradio_app import ChatController, convert_history_to_messages
from tests.fakes import FakeMemory, ScriptedModel
from tests.test_graph import plan


def make_service(responses, **kwargs):
    model = ScriptedModel(responses)
    memory = FakeMemory()
    deps = AgentDependencies(model, lambda: memory)
    service = AgentService(
        graph_factory=partial(create_agent_graph, deps), llm_factory=model, **kwargs
    )
    return service, model, memory


def test_api_simple_answer_is_completed_and_health_works():
    service, _, memory = make_service(
        [
            AIMessage(content='{"is_complex": false}'),
            AIMessage(content="안녕!"),
            AIMessage(content="{}"),
        ]
    )
    app = create_app(
        service=service, memory_factory=lambda: memory, mount_ui=False, initialize_memory=False
    )
    with TestClient(app) as client:
        response = client.post("/chat/start", json={"query": "안녕"})
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert response.json()["final_guide"] == "안녕!"
        assert client.get("/health").json()["tools_count"] == 12


def test_api_explicit_actions_are_not_reclassified_and_modified_plan_requires_approval():
    service, model, memory = make_service(
        [
            AIMessage(content='{"is_complex": true}'),
            plan("기존 작업"),
            AIMessage(content='[{"description": "수정 작업"}]'),
            AIMessage(content="추천 결과"),
            AIMessage(content="가이드"),
            AIMessage(content="{}"),
        ]
    )
    app = create_app(
        service=service, memory_factory=lambda: memory, mount_ui=False, initialize_memory=False
    )
    with TestClient(app) as client:
        start = client.post("/chat/start", json={"query": "요청", "user_id": "alice"}).json()
        thread = start["thread_id"]
        assert start["status"] == "pending_approval"
        request = {"thread_id": thread, "user_id": "alice", "action": "modify", "feedback": "수정"}
        revised = client.post("/chat/approve", json=request).json()
        assert revised["status"] == "pending_approval"
        assert revised["plan"][0]["description"] == "수정 작업"
        assert len(model.invocations) == 3
        completed = client.post(
            "/chat/approve",
            json={
                "thread_id": thread,
                "user_id": "alice",
                "action": "approve",
            },
        ).json()
        assert completed["final_guide"] == "가이드"
        assert completed["status"] == "completed"
        assert (
            client.post(
                "/chat/approve",
                json={
                    "thread_id": thread,
                    "user_id": "alice",
                    "action": "approve",
                },
            ).status_code
            == 404
        )


@pytest.mark.parametrize(
    "body",
    [
        {"thread_id": "x", "action": "surprise"},
        {"thread_id": "x"},
        {"thread_id": "x", "action": "modify"},
        {"thread_id": "x", "action": "modify", "feedback": "  "},
    ],
)
def test_api_rejects_invalid_approval_requests_before_execution(body):
    service, _, memory = make_service([])
    app = create_app(
        service=service, memory_factory=lambda: memory, mount_ui=False, initialize_memory=False
    )
    with TestClient(app) as client:
        assert client.post("/chat/approve", json=body).status_code == 422


def test_sessions_have_owner_conflict_capacity_and_expiration_checks():
    now = [0]
    service, _, _ = make_service(
        [
            AIMessage(content='{"is_complex": true}'),
            plan("작업"),
            AIMessage(content='{"is_complex": true}'),
            plan("다음 작업"),
        ],
        max_sessions=1,
        ttl_seconds=10,
        clock=lambda: now[0],
    )
    first = service.start("요청", "alice", thread_id="reserved")
    with pytest.raises(SessionNotFound):
        service.resume(first.thread_id, "bob", "approve")
    with pytest.raises(SessionConflict):
        service.start("새 요청", "alice", thread_id="reserved")
    with pytest.raises(SessionCapacityExceeded):
        service.start("새 요청", "alice")
    now[0] = 11
    with pytest.raises(SessionNotFound):
        service.resume(first.thread_id, "alice", "approve")
    assert service.start("새 요청", "alice").status == "pending_approval"


def test_concurrent_approval_runs_only_once():
    entered, release = Event(), Event()

    def delayed_recommendation(messages):
        entered.set()
        assert release.wait(timeout=5), "Test must release the first approval"
        return AIMessage(content="추천")

    service, _, _ = make_service(
        [
            AIMessage(content='{"is_complex": true}'),
            plan("작업"),
            delayed_recommendation,
            AIMessage(content="가이드"),
            AIMessage(content="{}"),
        ]
    )
    started = service.start("요청", "alice")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.resume, started.thread_id, "alice", "approve")
        try:
            assert entered.wait(timeout=5)
            with pytest.raises(SessionConflict):
                service.resume(started.thread_id, "alice", "approve")
        finally:
            release.set()
        assert first.result(timeout=5).status == "completed"


def test_failed_execution_returns_generic_error_and_releases_session():
    service, _, memory = make_service([RuntimeError("secret upstream details")])
    app = create_app(
        service=service, memory_factory=lambda: memory, mount_ui=False, initialize_memory=False
    )
    with TestClient(app) as client:
        response = client.post("/chat/start", json={"query": "요청", "thread_id": "failed"})
        assert response.status_code == 502
        assert "secret upstream details" not in response.text
        assert not service.has_pending("failed", "default_user")


def test_ui_unknown_intent_preserves_review_and_new_chat_discards_pending_graph():
    service, _, _ = make_service(
        [
            AIMessage(content='{"is_complex": true}'),
            plan("작업"),
            AIMessage(content="unparseable"),
        ]
    )
    controller = ChatController(service)
    history, thread, _ = controller.process_message("요청", [], "alice", None)
    assert len(history) == 2
    history, next_thread, _ = controller.process_message("음...", history, "alice", thread)
    assert next_thread == thread
    assert "명확히" in history[-1]["content"]
    assert controller.clear_chat(thread, "alice")[1] is None
    assert not service.has_pending(thread, "alice")


def test_ui_accepts_gradio6_text_blocks_as_chat_history():
    messages = convert_history_to_messages(
        [
            {"role": "user", "content": [{"type": "text", "text": "질문"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "답변"}]},
        ]
    )
    assert [item.content for item in messages] == ["질문", "답변"]


def test_gradio_mount_does_not_shadow_the_api_routes():
    service, _, memory = make_service([])
    app = create_app(
        service=service, memory_factory=lambda: memory, mount_ui=True, initialize_memory=False
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/").status_code == 200
