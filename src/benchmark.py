from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return recall score (0.0 to 1.0) based on presence of expected facts."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in ans_lower)
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score combining fact coverage and response structure."""
    if not answer.strip():
        return 0.0

    r_score = recall_points(answer, expected)
    # Check if response is well-structured (bullets or concise informative sentences)
    has_structure = 0.1 if ("\n-" in answer or "- " in answer or "\n•" in answer) else 0.05
    length_penalty = 0.0
    if len(answer) > 800:
        length_penalty = 0.1  # penalize verbose rambling
    elif len(answer) < 15 and expected:
        length_penalty = 0.2  # too curt

    quality = (r_score * 0.85) + has_structure - length_penalty
    return round(max(0.0, min(1.0, quality)), 2)


def run_agent_benchmark(agent_name: str, agent: Any, conversations: list[dict[str, Any]], config: LabConfig) -> BenchmarkRow:
    """Evaluate one agent over conversations and recall queries."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    total_compactions = 0
    all_users = set()

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        all_users.add(user_id)
        turns = conv.get("turns", [])
        recall_questions = conv.get("recall_questions", [])

        # 1. Main conversation thread
        main_thread = f"{conv_id}-main"
        for turn in turns:
            res = agent.reply(user_id, main_thread, turn)
            total_agent_tokens += res.get("tokens", 0)
            total_prompt_tokens += res.get("prompt_tokens", 0)

        total_compactions += agent.compaction_count(main_thread)

        # 2. Recall questions evaluated in new threads
        for q_idx, q_item in enumerate(recall_questions):
            recall_thread = f"{conv_id}-recall-{q_idx}"
            question_text = q_item["question"]
            expected = q_item.get("expected_contains", [])

            res = agent.reply(user_id, recall_thread, question_text)
            total_agent_tokens += res.get("tokens", 0)
            total_prompt_tokens += res.get("prompt_tokens", 0)
            total_compactions += agent.compaction_count(recall_thread)

            reply = res.get("reply", "")
            r_pts = recall_points(reply, expected)
            q_pts = heuristic_quality(reply, expected)

            recall_scores.append(r_pts)
            quality_scores.append(q_pts)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    # Calculate memory growth
    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        memory_growth = sum(agent.memory_file_size(u) for u in all_users)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 2),
        response_quality=round(avg_quality, 2),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append([
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality * 100:.1f}%",
            f"{r.memory_growth_bytes:,} B",
            r.compactions,
        ])
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both Standard Benchmark and Long-Context Stress Benchmark."""
    base_dir = Path(__file__).resolve().parent.parent
    config = load_config(base_dir)

    conv_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    print("=" * 80)
    print("PHASE 2, TRACK 3, DAY 17: MEMORY SYSTEMS BENCHMARK")
    print("=" * 80)

    # 1. Standard Benchmark
    print("\n[1] Running Standard Benchmark (data/conversations.json)...")
    std_convs = load_conversations(conv_path)

    # Clean benchmark state dir for fresh run
    std_state_dir = config.base_dir / "state" / "std_bench"
    if std_state_dir.exists():
        shutil.rmtree(std_state_dir)
    std_state_dir.mkdir(parents=True, exist_ok=True)
    std_config = LabConfig(
        base_dir=config.base_dir,
        data_dir=config.data_dir,
        state_dir=std_state_dir,
        compact_threshold_tokens=config.compact_threshold_tokens,
        compact_keep_messages=config.compact_keep_messages,
        model=config.model,
        judge_model=config.judge_model,
    )

    baseline_std = BaselineAgent(std_config, force_offline=True)
    advanced_std = AdvancedAgent(std_config, force_offline=True)

    row_base_std = run_agent_benchmark("Baseline", baseline_std, std_convs, std_config)
    row_adv_std = run_agent_benchmark("Advanced", advanced_std, std_convs, std_config)

    print("\n### Standard Benchmark Results")
    print(format_rows([row_base_std, row_adv_std]))

    # 2. Long-Context Stress Benchmark
    print("\n[2] Running Long-Context Stress Benchmark (data/advanced_long_context.json)...")
    stress_convs = load_conversations(stress_path)

    stress_state_dir = config.base_dir / "state" / "stress_bench"
    if stress_state_dir.exists():
        shutil.rmtree(stress_state_dir)
    stress_state_dir.mkdir(parents=True, exist_ok=True)
    stress_config = LabConfig(
        base_dir=config.base_dir,
        data_dir=config.data_dir,
        state_dir=stress_state_dir,
        compact_threshold_tokens=config.compact_threshold_tokens,
        compact_keep_messages=config.compact_keep_messages,
        model=config.model,
        judge_model=config.judge_model,
    )

    baseline_stress = BaselineAgent(stress_config, force_offline=True)
    advanced_stress = AdvancedAgent(stress_config, force_offline=True)

    row_base_stress = run_agent_benchmark("Baseline", baseline_stress, stress_convs, stress_config)
    row_adv_stress = run_agent_benchmark("Advanced", advanced_stress, stress_convs, stress_config)

    print("\n### Long-Context Stress Benchmark Results")
    print(format_rows([row_base_stress, row_adv_stress]))
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
