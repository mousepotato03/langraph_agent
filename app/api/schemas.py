"""Validated API requests and explicit conversation statuses."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    user_id: str = Field(default="default_user", min_length=1, max_length=200)
    thread_id: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def nonblank(self):
        if not self.query.strip() or not self.user_id.strip():
            raise ValueError("query and user_id must not be blank")
        return self


class ApproveRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=200)
    user_id: str = Field(default="default_user", min_length=1, max_length=200)
    action: Literal["approve", "modify", "cancel"]
    feedback: str | None = Field(default=None, max_length=10000)

    @model_validator(mode="after")
    def modification_requires_feedback(self):
        if self.action == "modify" and (not self.feedback or not self.feedback.strip()):
            raise ValueError("modify requires nonblank feedback")
        return self


class TaskItem(BaseModel):
    id: str
    description: str


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["pending_approval", "completed", "cancelled"]
    message: str
    is_complex: bool = False
    plan: list[TaskItem] | None = None
    # Kept for API compatibility: contains the final answer from either branch.
    final_guide: str | None = None


class HealthResponse(BaseModel):
    status: str
    tools_count: int
    profiles_count: int
