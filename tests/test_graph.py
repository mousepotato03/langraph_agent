import json
from functools import partial

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from agent.dependencies import AgentDependencies
from agent.graph import create_agent_graph, create_initial_state
from app.service import AgentService
from core.config import GRAPH_RECURSION_LIMIT
from tests.fakes import FakeMemory, ScriptedModel
from tools.registry import execute_tool


def call(name, identifier, **args):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": identifier, "type": "tool_call"}]
    )


def plan(*tasks):
    return AIMessage(
        content=json.dumps(
            {
                "analysis": "작업을 나누어 추천합니다.",
                "subtasks": [{"description": task} for task in tasks],
            },
            ensure_ascii=False,
        )
    )


def build(responses, runner=execute_tool):
    model = ScriptedModel(responses)
    memory = FakeMemory()
    dependencies = AgentDependencies(model, lambda: memory, runner)
    graph = create_agent_graph(dependencies)
    return graph, model, memory, dependencies


def config():
    return {"configurable": {"thread_id": "test"}, "recursion_limit": GRAPH_RECURSION_LIMIT}


def test_simple_math_real_graph_and_reflection_use_the_final_answer():
    graph, model, memory, _ = build(
        [
            AIMessage(content='{"is_complex": false}'),
            call("calculate_math", "math-1", expression="22 * 34"),
            AIMessage(content="22 × 34 = 748입니다."),
            AIMessage(content='{"interests": ["계산"]}'),
        ]
    )
    result = graph.invoke(create_initial_state("22 * 34 알려줘", "alice"), config())
    assert result["final_answer"] == "22 × 34 = 748입니다."
    tool_result = next(item for item in result["messages"] if isinstance(item, ToolMessage))
    assert tool_result.content == "22 * 34 = 748"
    assert tool_result.tool_call_id == "math-1"
    assert "748" in model.invocations[-1][-1].content
    assert memory.saved[0][0] == "alice"
    assert not graph.get_state(config()).next
    assert all(binding[1]["parallel_tool_calls"] is False for binding in model.bindings)


def test_modified_plan_interrupts_again_until_explicitly_approved():
    calls = []

    def search(name, args):
        calls.append((name, args))
        return json.dumps(
            {
                "recommended_tool": {
                    "name": "Example AI",
                    "description": "테스트 도구",
                    "scores": {"final_score": 0.8},
                },
                "should_fallback": False,
            }
        )

    graph, model, _, _ = build(
        [
            AIMessage(content='{"is_complex": true}'),
            plan("유료 도구 조사", "영상 제작"),
            AIMessage(content='[{"description": "무료 도구로 영상 제작"}]'),
            call("retrieve_docs", "search-1", query="free video"),
            AIMessage(content="무료 도구 사용 가이드"),
            AIMessage(content="{}"),
        ],
        search,
    )
    first = graph.invoke(create_initial_state("영상을 만들고 싶어"), config())
    assert first["__interrupt__"][0].value["sub_tasks"] == ["유료 도구 조사", "영상 제작"]
    assert calls == []
    modified = graph.invoke(Command(resume={"action": "modify", "feedback": "무료만"}), config())
    assert modified["__interrupt__"][0].value["sub_tasks"] == ["무료 도구로 영상 제작"]
    assert calls == []
    assert len(model.invocations) == 3
    result = graph.invoke(Command(resume={"action": "approve"}), config())
    assert result["final_guide"] == "무료 도구 사용 가이드"
    assert len(calls) == 1
    assert not graph.get_state(config()).next


@pytest.mark.parametrize(
    "decision",
    [
        {"action": "clarify"},
        {"action": "unexpected"},
        {"action": "modify", "feedback": ""},
        "approve",
    ],
)
def test_invalid_decision_keeps_gate_closed(decision):
    graph, model, memory, _ = build(
        [
            AIMessage(content='{"is_complex": true}'),
            plan("영상 제작"),
        ],
        lambda *_: pytest.fail("Approval gate must prevent tool calls"),
    )
    graph.invoke(create_initial_state("영상을 만들어줘"), config())
    result = graph.invoke(Command(resume=decision), config())
    assert result["__interrupt__"]
    assert result["approval_status"] == "pending"
    assert len(model.invocations) == 2
    assert not memory.saved


@pytest.mark.parametrize(
    "revision",
    [
        AIMessage(content="not JSON"),
        AIMessage(content="[]"),
        RuntimeError("model unavailable"),
    ],
)
def test_failed_revision_preserves_old_plan_and_requires_review(revision):
    graph, _, _, _ = build(
        [
            AIMessage(content='{"is_complex": true}'),
            plan("원래 작업"),
            revision,
        ]
    )
    graph.invoke(create_initial_state("요청"), config())
    result = graph.invoke(Command(resume={"action": "modify", "feedback": "무료만"}), config())
    assert result["sub_tasks"] == ["원래 작업"]
    assert result["approval_status"] == "pending"
    assert result["__interrupt__"]


def test_cancel_does_not_call_a_model_or_save_preferences():
    graph, model, memory, _ = build(
        [
            AIMessage(content='{"is_complex": true}'),
            plan("영상 제작"),
        ]
    )
    graph.invoke(create_initial_state("요청"), config())
    result = graph.invoke(Command(resume={"action": "cancel"}), config())
    assert result["approval_status"] == "cancelled"
    assert len(model.invocations) == 2
    assert not memory.saved
    assert not graph.get_state(config()).next


def test_each_task_keeps_its_own_result_and_does_not_mutate_earlier_state():
    calls = []

    def search(name, args):
        calls.append(args["query"])
        name = {"first": "Writer", "second": "Video"}[args["query"]]
        return json.dumps(
            {
                "recommended_tool": {
                    "name": name,
                    "description": name,
                    "scores": {"final_score": 0.9},
                },
                "should_fallback": False,
            }
        )

    graph, model, _, _ = build(
        [
            AIMessage(content='{"is_complex": true}'),
            plan("스크립트 작성", "영상 생성"),
            call("retrieve_docs", "first-call", query="first"),
            call("retrieve_docs", "second-call", query="second"),
            AIMessage(content="최종 가이드"),
            AIMessage(content="{}"),
        ],
        search,
    )
    paused = graph.invoke(create_initial_state("요청"), config())
    old_recommendations = paused["tool_recommendations"]
    result = graph.invoke(Command(resume={"action": "approve"}), config())
    assert "Writer" in result["tool_recommendations"]["task_1"]
    assert "Video" in result["tool_recommendations"]["task_2"]
    assert "Writer" not in result["tool_recommendations"]["task_2"]
    assert old_recommendations == {}
    # The second recommendation call has no evidence or tool pairs from the first task.
    assert not any(isinstance(item, ToolMessage) for item in model.invocations[3])
    assert not result["task_messages"] and not result["task_documents"]
    assert calls == ["first", "second"]


def test_weak_retrieval_continues_to_web_and_passes_observations_back_to_model():
    def search(name, args):
        if name == "retrieve_docs":
            return json.dumps(
                {
                    "recommended_tool": {"name": "Weak candidate", "scores": {"final_score": 0.2}},
                    "should_fallback": True,
                }
            )
        return json.dumps({"results": [{"name": "Web candidate", "url": "https://example.org"}]})

    graph, model, _, _ = build(
        [
            AIMessage(content='{"is_complex": true}'),
            plan("작업"),
            call("retrieve_docs", "rag", query="video"),
            call("google_search_tool", "web", query="video"),
            AIMessage(content="Web candidate를 추천합니다."),
            AIMessage(content="최종 가이드"),
            AIMessage(content="{}"),
        ],
        search,
    )
    graph.invoke(create_initial_state("요청"), config())
    result = graph.invoke(Command(resume={"action": "approve"}), config())
    assert result["tool_recommendations"]["task_1"] == "Web candidate를 추천합니다."
    assert "Web candidate" in model.invocations[4][-1].content


def test_service_reports_simple_completion_without_creating_pending_session():
    _, model, memory, dependencies = build(
        [
            AIMessage(content='{"is_complex": false}'),
            AIMessage(content="안녕하세요."),
            AIMessage(content="{}"),
        ]
    )
    service = AgentService(
        graph_factory=partial(create_agent_graph, dependencies), llm_factory=model
    )
    result = service.start("안녕", "alice")
    assert result.status == "completed"
    assert result.final_guide == "안녕하세요."
    assert not service.has_pending(result.thread_id, "alice")
    assert memory.saved


def test_reflection_failure_does_not_discard_a_successful_answer():
    graph, _, _, _ = build(
        [
            AIMessage(content='{"is_complex": false}'),
            AIMessage(content="최종 답변"),
            RuntimeError("memory model failed"),
        ]
    )
    result = graph.invoke(create_initial_state("요청"), config())
    assert result["final_answer"] == "최종 답변"


def test_five_tasks_with_three_tool_calls_each_fit_the_service_recursion_limit():
    responses = [AIMessage(content='{"is_complex": true}'), plan(*[f"작업 {i}" for i in range(5)])]
    for task in range(5):
        responses.extend(
            call("calculate_math", f"math-{task}-{step}", expression="1+2") for step in range(3)
        )
        responses.append(AIMessage(content=f"작업 {task} 추천"))
    responses.extend([AIMessage(content="모든 작업 가이드"), AIMessage(content="{}")])
    graph, model, _, _ = build(responses)
    graph.invoke(create_initial_state("요청"), config())
    result = graph.invoke(Command(resume={"action": "approve"}), config())
    assert result["final_guide"] == "모든 작업 가이드"
    assert len(result["tool_recommendations"]) == 5
    assert len([item for item in result["messages"] if isinstance(item, ToolMessage)]) == 15
    assert not model.responses


def test_reflection_preserves_preferences_written_by_a_tool_during_the_run():
    def runner(name, args):
        assert name == "write_memory"
        memory.save_user_profile(args["user_id"], {"interests": ["도구에서 저장한 관심사"]})
        return '{"status": "success"}'

    graph, _, memory, _ = build(
        [
            AIMessage(content='{"is_complex": true}'),
            plan("작업"),
            call("write_memory", "write", user_id="alice", preferences='{"interests": ["영상"]}'),
            AIMessage(content="추천"),
            AIMessage(content="가이드"),
            AIMessage(content="{}"),
        ],
        runner,
    )
    graph.invoke(create_initial_state("요청", "alice"), config())
    graph.invoke(Command(resume={"action": "approve"}), config())
    assert memory.profiles["alice"]["interests"] == ["도구에서 저장한 관심사"]
