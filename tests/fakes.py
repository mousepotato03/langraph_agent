"""Network-free adapters; LangGraph itself is real in the integration tests."""

from collections import deque

from langchain_core.messages import AIMessage, ToolMessage


def assert_tool_protocol(messages):
    pending = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in pending, "Tool result must match a pending call"
            pending.remove(message.tool_call_id)
        else:
            assert not pending, "Every call must receive a result before the next message"
            if isinstance(message, AIMessage):
                pending.update(call["id"] for call in message.tool_calls)
    assert not pending, "Model invocation must contain complete tool-call/result pairs"


class ScriptedModel:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.invocations = []
        self.bindings = []

    def __call__(self, **kwargs):
        return self

    def bind_tools(self, tools, **kwargs):
        self.bindings.append((tuple(tool.name for tool in tools), kwargs))
        return self

    def invoke(self, messages):
        assert_tool_protocol(messages)
        self.invocations.append(messages)
        assert self.responses, "Unexpected model call"
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response(messages) if callable(response) else response


class FakeMemory:
    def __init__(self):
        self.profiles = {}
        self.saved = []

    def load_user_profile(self, user_id):
        return self.profiles.get(user_id)

    def save_user_profile(self, user_id, preferences):
        self.saved.append((user_id, preferences))
        self.profiles[user_id] = preferences
        return True

    def get_tools_count(self):
        return 12

    def get_profiles_count(self):
        return len(self.profiles)
