"""Gradio rendering and a testable controller; graph execution lives in AgentService."""

import logging
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.service import AgentService, ServiceError
from core.memory import get_memory_manager

logger = logging.getLogger(__name__)


def convert_history_to_messages(history: list[dict]) -> list[BaseMessage]:
    messages = []
    for item in history:
        content = item.get("content", "")
        # Gradio 6 preprocesses text into typed content blocks.
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if not isinstance(content, str):
            continue
        if item.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif item.get("role") == "assistant":
            messages.append(AIMessage(content=content))
    return messages


class ChatController:
    def __init__(self, service: AgentService):
        self.service = service

    def process_message(
        self, message: str, history: list | None, user_id: str, thread_id: str | None
    ):
        history = list(history or [])
        if not message.strip():
            return history, thread_id, "메시지를 입력해주세요."
        try:
            if thread_id:
                result = self.service.respond(thread_id, user_id, message)
            else:
                result = self.service.start(
                    message, user_id, chat_history=convert_history_to_messages(history)
                )
            text = result.final_guide or result.message
            next_thread = result.thread_id if result.status == "pending_approval" else None
            status = {
                "pending_approval": "계획 검토 중 — 승인·수정·취소를 알려주세요.",
                "completed": "완료",
                "cancelled": "취소됨",
            }[result.status]
        except (ServiceError, ValueError) as error:
            logger.warning("UI request failed", exc_info=True)
            text = str(error)
            next_thread = (
                thread_id if thread_id and self.service.has_pending(thread_id, user_id) else None
            )
            status = "요청을 처리하지 못했습니다."
        history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": text},
            ]
        )
        return history, next_thread, status

    def clear_chat(self, thread_id: str | None, user_id: str):
        try:
            self.service.discard(thread_id, user_id)
        except ServiceError as error:
            # Keep the pending thread so a concurrent request is not abandoned.
            return None, thread_id, str(error)
        return [], None, "새 대화를 시작합니다."


def create_gradio_ui(service: AgentService, memory_factory=get_memory_manager):
    import gradio as gr

    controller = ChatController(service)
    with gr.Blocks(title="AI 101 - AI 도구 추천 에이전트") as demo:
        current_thread = gr.State(None)
        gr.Markdown(
            "# AI 101\n원하는 작업을 설명하면 AI 도구를 추천하고 사용 가이드를 작성합니다.\n\n"
            "복잡한 요청은 계획을 먼저 검토합니다. 수정한 계획은 다시 승인해주세요."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="대화", height=500)
                with gr.Row():
                    message = gr.Textbox(
                        placeholder="예: 무료 도구로 유튜브 쇼츠를 만들고 싶어",
                        show_label=False,
                        scale=4,
                    )
                    submit = gr.Button("전송", variant="primary")
            with gr.Column(scale=1):
                user_id = gr.Textbox(
                    label="사용자 ID",
                    value=lambda: f"gradio_{uuid4().hex}",
                    info="같은 ID를 사용하면 저장된 선호도를 이어서 사용합니다.",
                )
                status = gr.Textbox(label="상태", value="대기 중", interactive=False)
                clear = gr.Button("새 대화")
                tools_count = gr.Number(label="등록된 도구", value=0, interactive=False)
                profiles_count = gr.Number(label="사용자 프로필", value=0, interactive=False)
                refresh = gr.Button("통계 새로고침")

        for event in (submit.click, message.submit):
            event(
                fn=controller.process_message,
                inputs=[message, chatbot, user_id, current_thread],
                outputs=[chatbot, current_thread, status],
            ).then(fn=lambda: "", outputs=message)

        def clear_chat(history, thread_id, identity):
            cleared, next_thread, text = controller.clear_chat(thread_id, identity)
            return history if cleared is None else cleared, next_thread, text

        clear.click(
            fn=clear_chat,
            inputs=[chatbot, current_thread, user_id],
            outputs=[chatbot, current_thread, status],
        )

        def refresh_stats():
            memory = memory_factory()
            return memory.get_tools_count(), memory.get_profiles_count()

        refresh.click(fn=refresh_stats, outputs=[tools_count, profiles_count])
        demo.load(fn=refresh_stats, outputs=[tools_count, profiles_count])
    return demo
