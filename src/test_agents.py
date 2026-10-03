from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated configuration for unit and behavioral tests."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    dummy_provider = ProviderConfig(
        provider="custom",
        model_name="mock-model",
        temperature=0.0,
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=80,  # low threshold so compaction triggers quickly
        compact_keep_messages=2,
        model=dummy_provider,
        judge_model=dummy_provider,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, read, edited, and queried for size."""
    profiles_dir = tmp_path / "profiles"
    store = UserProfileStore(root_dir=profiles_dir)

    user_id = "test_user"
    initial_text = "# User Profile\n\n- Name: Alice\n- Location: Da Nang\n"

    # 1. Write text
    written_path = store.write_text(user_id, initial_text)
    assert written_path.exists()
    assert written_path.name == "User.md"

    # 2. Read text
    read_back = store.read_text(user_id)
    assert "Alice" in read_back
    assert "Da Nang" in read_back

    # 3. Edit text
    changed = store.edit_text(user_id, "Da Nang", "Hue")
    assert changed is True
    updated_text = store.read_text(user_id)
    assert "Hue" in updated_text
    assert "Da Nang" not in updated_text

    # 4. File size
    assert store.file_size(user_id) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction and summarize older context."""
    manager = CompactMemoryManager(threshold_tokens=60, keep_messages=2)
    thread_id = "test-thread-01"

    # Append turns that exceed 60 tokens
    manager.append(thread_id, "user", "Tin tức Artemis III và mục tiêu phóng năm 2027 rất quan trọng cho lộ trình không gian dài hạn.")
    manager.append(thread_id, "assistant", "Tôi đã ghi nhận thông tin về Artemis III và các mốc kiểm thử quỹ đạo.")
    manager.append(thread_id, "user", "Chiếc máy bay X-59 đã đạt vận tốc siêu thanh Mach 1.1 ở độ cao 29500 feet để giảm tiếng nổ.")
    manager.append(thread_id, "assistant", "X-59 là ví dụ tuyệt vời về việc tối ưu hiệu năng nhưng vẫn giảm ngoại tác âm thanh.")

    assert manager.compaction_count(thread_id) > 0
    ctx = manager.context(thread_id)
    assert ctx["summary"] != ""
    assert len(ctx["messages"]) <= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced agent remembers across sessions while baseline agent does not."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "dungct_test"

    # Session 1: User introduces themselves
    intro_message = "Chào bạn, mình tên là DũngCT. Mình thích Python và đồ uống yêu thích là cà phê sữa đá."
    baseline.reply(user_id, "thread-session-1", intro_message)
    advanced.reply(user_id, "thread-session-1", intro_message)

    # Session 2: Fresh thread asking recall question
    recall_question = "Mình tên gì và đồ uống yêu thích là gì?"
    base_res = baseline.reply(user_id, "thread-session-2", recall_question)
    adv_res = advanced.reply(user_id, "thread-session-2", recall_question)

    # Baseline has no cross-session memory
    assert "DũngCT" not in base_res["reply"]
    assert "cà phê sữa đá" not in base_res["reply"]

    # Advanced agent recalls from persistent User.md
    assert "DũngCT" in adv_res["reply"]
    assert "cà phê sữa đá" in adv_res["reply"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread with compaction."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "stress_user"
    thread_id = "long-thread-01"

    long_messages = [
        "Đây là đoạn văn dài thứ nhất về NASA Artemis III công bố prime crew cho nhiệm vụ bay quanh Mặt Trăng năm 2027.",
        "Tiếp theo là đoạn văn dài thứ hai phân tích về hệ thống hỗ trợ sự sống và module Orion trong không gian sâu.",
        "Đoạn văn dài thứ ba nói về chuyến bay thử nghiệm X-59 với mục tiêu giảm thiểu tiếng ồn siêu thanh đối với mặt đất.",
        "Đoạn văn thứ tư đề cập đến báo cáo WMO về hiện tượng El Nino với xác suất trên 80 phần trăm vào cuối năm 2026.",
        "Đoạn văn thứ năm trình bày kế hoạch năng lượng sạch của British Columbia với mức tiết kiệm điện cho hàng trăm ngàn hộ dân.",
        "Đoạn văn thứ sáu tóm tắt các luận điểm vận hành và hệ thống bài học thực tiễn rút ra từ các trường hợp trên.",
        "Đoạn văn thứ bảy so sánh chi phí mở rộng hạ tầng và tối ưu hóa nhu cầu sử dụng thực tế trong quản lý dự án.",
    ]

    for msg in long_messages:
        baseline.reply(user_id, thread_id, msg)
        advanced.reply(user_id, thread_id, msg)

    base_prompt_tokens = baseline.prompt_token_usage(thread_id)
    adv_prompt_tokens = advanced.prompt_token_usage(thread_id)

    # Advanced agent must trigger compaction and keep prompt context smaller than baseline
    assert advanced.compaction_count(thread_id) > 0
    assert adv_prompt_tokens < base_prompt_tokens
