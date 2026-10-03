from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent.

    Characteristics:
    - Within-session (in-thread) memory only
    - No persistent User.md
    - Forgets all long-term facts when switched to a new thread
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Process a turn and return the assistant reply with token accounting."""
        if self.langchain_agent is not None and not self.force_offline:
            try:
                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                output_text = result["messages"][-1].content
                reply_tokens = estimate_tokens(output_text)
                prompt_tokens = estimate_tokens(message)
                session = self.sessions.setdefault(thread_id, SessionState())
                session.token_usage += reply_tokens
                session.prompt_tokens_processed += prompt_tokens
                return {"reply": output_text, "tokens": reply_tokens, "prompt_tokens": prompt_tokens}
            except Exception:
                # Fallback to offline deterministic mode
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent-generated tokens for this thread."""
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt context tokens processed for this thread."""
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compaction mechanism."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Offline deterministic reply path without long-term memory."""
        session = self.sessions.setdefault(thread_id, SessionState())

        # Baseline carries the entire uncompressed thread history in prompt
        prior_context_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        turn_prompt_tokens = prior_context_tokens + estimate_tokens(message)
        session.prompt_tokens_processed += turn_prompt_tokens

        # Record incoming message
        session.messages.append({"role": "user", "content": message})

        # Generate response based strictly on within-thread messages
        text_lower = message.lower()
        if "?" in message or any(kw in text_lower for kw in ["nhắc lại", "là ai", "ở đâu", "nghề gì", "đồ uống", "món ăn"]):
            # Check if this thread has the answer
            in_thread_text = " ".join(m["content"] for m in session.messages[:-1])
            if in_thread_text:
                reply_text = f"Trong phiên trò chuyện này, tôi thấy bạn đã đề cập: {in_thread_text[:100]}..."
            else:
                # In a new session, baseline agent knows nothing!
                reply_text = "Chào bạn! Tôi là trợ lý AI. Hiện tại trong phiên này tôi chưa có thông tin về bạn."
        else:
            reply_text = "Chào bạn, tôi đã ghi nhận thông tin trong phiên làm việc này."

        reply_tokens = estimate_tokens(reply_text)
        session.token_usage += reply_tokens
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "reply": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": turn_prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally build a LangChain/LangGraph agent with InMemorySaver."""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            model = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            return create_react_agent(model, tools=[], checkpointer=checkpointer)
        except Exception:
            return None
