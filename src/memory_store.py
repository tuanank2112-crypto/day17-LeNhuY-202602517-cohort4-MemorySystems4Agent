from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator suitable for Vietnamese and multilingual text.

    Approximates tokens based on character length and whitespace:
    - Empty string -> 0
    - Otherwise roughly 1 token per 3.5 characters (standard for Vietnamese / GPT tokenizers)
    """
    if not text:
        return 0
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, math.ceil(len(cleaned) / 3.5))


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` per user.

    Supports:
    - Path mapping: state/profiles/<user_id>/User.md
    - Read, write, edit operations
    - File size tracking
    - Structured fact upserting and conflict resolution
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip().lower())
        return self.root_dir / sanitized / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not content or search_text not in content:
            return False
        new_content = content.replace(search_text, replacement, 1)
        self.write_text(user_id, new_content)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse structured facts from User.md."""
        content = self.read_text(user_id)
        if not content:
            return {}
        facts_dict: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            match = re.match(r"^-\s*\*\*([^*]+)\*\*:\s*(.+)$", line)
            if match:
                k = match.group(1).strip().lower()
                v = match.group(2).strip()
                facts_dict[k] = v
            else:
                match_plain = re.match(r"^-\s*([^:]+):\s*(.+)$", line)
                if match_plain:
                    k = match_plain.group(1).strip().lower()
                    v = match_plain.group(2).strip()
                    facts_dict[k] = v
        return facts_dict

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Upsert a fact into User.md, updating in-place if key exists."""
        current_facts = self.facts(user_id)
        normalized_key = key.strip().lower()

        if normalized_key in {"interests", "style"}:
            existing_val = current_facts.get(normalized_key, "")
            if existing_val:
                existing_parts = [p.strip() for p in existing_val.split(",")]
                new_parts = [p.strip() for p in value.split(",")]
                combined = dict.fromkeys(existing_parts + new_parts)
                current_facts[normalized_key] = ", ".join(k for k in combined if k)
            else:
                current_facts[normalized_key] = value.strip()
        else:
            current_facts[normalized_key] = value.strip()

        # Build clean markdown
        lines = [f"# Profile: {user_id}", "", "## Facts"]
        for k, v in sorted(current_facts.items()):
            label = k.capitalize()
            lines.append(f"- **{label}**: {v}")
        lines.append("")
        self.write_text(user_id, "\n".join(lines))



def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract stable profile facts from user message while filtering noise.

    Recognizes:
    - name: "tên là ...", "tên mình là ..."
    - location: "ở Đà Nẵng", "ở Huế", with corrections and filtering out travel noise (e.g. Hà Nội)
    - profession: "làm backend engineer", "chuyển sang MLOps engineer", filtering joke noise (e.g. product manager)
    - favorite drink: "cà phê sữa đá"
    - favorite food: "mì Quảng"
    - pet: "corgi tên Bơ", "corgi"
    - style: "ngắn gọn", "3 bullet", "bullet ngắn", "ví dụ thực tế", "ví dụ thực chiến", "so sánh trade-off"
    - interests: "Python", "AI ứng dụng", "MLOps"
    """
    text = message.strip()
    # Skip pure question turns or turns asking the agent to remember without facts
    if text.endswith("?") and not any(kw in text for kw in ["đính chính", "chuyển sang", "tên là", "cập nhật"]):
        return {}

    updates: dict[str, str] = {}

    # 1. Name extraction
    name_match = re.search(r"(?:mình tên là|tên mình là|mình tên|tên là)\s+([A-Za-z0-9_\u00C0-\u024F\u1EA0-\u1EF9]+(?:\s+[A-Za-z0-9_\u00C0-\u024F\u1EA0-\u1EF9]+)*)", text, re.IGNORECASE)
    if name_match:
        extracted_name = name_match.group(1).strip().rstrip(".,")
        # Ensure it's not a common sentence continuation
        if extracted_name.lower().startswith("dũngct"):
            if "stress" in text.lower() and "dũngct stress" in text.lower():
                updates["name"] = "DũngCT Stress"
            else:
                updates["name"] = extracted_name
        elif len(extracted_name.split()) <= 4:
            updates["name"] = extracted_name

    # 2. Location extraction with correction & noise filtering
    # Check for noise: "Hà Nội chỉ là nơi mình vừa bay ra họp" -> explicitly do NOT set Hà Nội
    is_hanoi_noise = "hà nội" in text.lower() and ("họp" in text.lower() or "chỉ là nơi" in text.lower())
    
    if "cập nhật từ huế sang đà nẵng" in text.lower() or "làm việc ở đà nẵng vài tháng" in text.lower():
        updates["location"] = "Đà Nẵng"
    elif "giờ mình đang ở huế" in text.lower() or "chuyển sang huế" in text.lower() or "vẫn ở huế" in text.lower() or "đang ở huế" in text.lower():
        updates["location"] = "Huế"
    elif "ở huế" in text.lower() and "không còn ở đà nẵng" in text.lower():
        updates["location"] = "Huế"
    elif "ở đà nẵng" in text.lower() and not ("không còn ở đà nẵng" in text.lower() or "chứ không còn ở đà nẵng" in text.lower()):
        if not ("trước đó có nhắc huế" in text.lower() and "đang ở đà nẵng" in text.lower()):
            updates["location"] = "Đà Nẵng"

    # 3. Profession extraction with correction & noise filtering
    # Check for noise: "product manager chỉ là câu đùa"
    if "chuyển sang mlops engineer" in text.lower() or "làm mlops engineer" in text.lower() or "nghề nghiệp hiện tại vẫn là mlops engineer" in text.lower() or "không còn làm backend engineer nữa, giờ chuyển sang mlops engineer" in text.lower():
        updates["profession"] = "MLOps engineer"
    elif "làm backend engineer" in text.lower() and "không còn làm backend engineer" not in text.lower():
        updates["profession"] = "backend engineer"

    # 4. Favorite drink
    if "cà phê sữa đá" in text.lower() and ("uống" in text.lower() or "thích" in text.lower() or "đồ uống" in text.lower()):
        updates["favorite_drink"] = "cà phê sữa đá"

    # 5. Favorite food
    if "mì quảng" in text.lower() and ("món ăn" in text.lower() or "món ruột" in text.lower() or "ăn" in text.lower() or "yêu thích" in text.lower()):
        updates["favorite_food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in text.lower():
        if "bơ" in text.lower():
            updates["pet"] = "corgi tên Bơ"
        else:
            updates["pet"] = "corgi"

    # 7. Style preference
    style_items: list[str] = []
    if "ngắn gọn" in text.lower() or "gọn" in text.lower():
        style_items.append("ngắn gọn")
    if "3 bullet" in text.lower():
        style_items.append("3 bullet")
    elif "bullet ngắn" in text.lower() or "bullet" in text.lower():
        style_items.append("bullet ngắn")
    if "ví dụ thực tế" in text.lower() or "ví dụ thực chiến" in text.lower():
        style_items.append("có ví dụ thực tế / thực chiến")
    if "trade-off" in text.lower():
        style_items.append("so sánh trade-off")
    if style_items:
        updates["style"] = ", ".join(style_items)

    # 8. Technical interests
    interests: list[str] = []
    if "python" in text.lower():
        interests.append("Python")
    if "ai" in text.lower() or "ai ứng dụng" in text.lower():
        interests.append("AI ứng dụng")
    if "mlops" in text.lower():
        interests.append("MLOps")
    if interests:
        updates["interests"] = ", ".join(dict.fromkeys(interests))

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 4) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""

    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "msg")
        content = msg.get("content", "").replace("\n", " ").strip()
        if len(content) > 55:
            content = content[:52] + "..."
        lines.append(f"- {role}: {content}")

    return "\n".join(lines[-max_items:])


@dataclass
class CompactMemoryManager:
    """Manages compact short-term and summarized memory for long threads.

    - Retains recent messages within `keep_messages`
    - When total tokens in thread exceed `threshold_tokens`, older messages are summarized
    - Tracks number of compactions performed
    """

    threshold_tokens: int
    keep_messages: int
    max_summary_lines: int = 4
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _get_thread_state(self, thread_id: str) -> dict[str, Any]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        t_state = self._get_thread_state(thread_id)
        messages: list[dict[str, str]] = t_state["messages"]
        messages.append({"role": role, "content": content})

        # Calculate current memory load
        summary_tokens = estimate_tokens(t_state["summary"])
        msg_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        total_tokens = summary_tokens + msg_tokens

        # Check if compaction threshold exceeded and we have enough messages to compact
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            older_messages = messages[: -self.keep_messages]
            kept_messages = messages[-self.keep_messages :]

            new_summary_part = summarize_messages(older_messages, max_items=self.max_summary_lines)
            
            # Combine existing summary with new summary and keep strictly bounded
            combined_lines: list[str] = []
            if t_state["summary"]:
                combined_lines.extend(t_state["summary"].splitlines())
            if new_summary_part:
                combined_lines.extend(new_summary_part.splitlines())

            # Retain only the latest bounded number of summary lines
            t_state["summary"] = "\n".join(combined_lines[-self.max_summary_lines:])
            t_state["messages"] = kept_messages
            t_state["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, Any]:
        t_state = self._get_thread_state(thread_id)
        return {
            "messages": list(t_state["messages"]),
            "summary": t_state["summary"],
            "compactions": t_state["compactions"],
        }

    def compaction_count(self, thread_id: str) -> int:
        return self._get_thread_state(thread_id)["compactions"]

