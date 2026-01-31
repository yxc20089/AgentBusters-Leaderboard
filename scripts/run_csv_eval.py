"""
Batch evaluation runner for CSV datasets.

Usage:
    python -m scripts.run_csv_eval \
      --dataset-path /app/data/public.csv \
      --simulation-date 2024-12-31 \
      --difficulty medium \
      --purple-endpoint http://localhost:8010 \
      --output /data/results/summary.json

Reads a CSV via CsvFinanceDatasetProvider, generates tasks with DynamicTaskGenerator,
runs ComprehensiveEvaluator with PurpleHTTPAgentClient, and writes per-task results plus
aggregated summary to the output JSON file.
"""

import argparse
import asyncio
import json
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, List

from cio_agent.a2a_client import PurpleHTTPAgentClient
from cio_agent.data_providers import CsvFinanceDatasetProvider
from cio_agent.evaluator import ComprehensiveEvaluator
from cio_agent.models import TaskDifficulty
from cio_agent.task_generator import DynamicTaskGenerator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate CSV finance tasks.")
    parser.add_argument("--dataset-path", required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--purple-endpoint",
        required=True,
        help="Purple Agent endpoint URL (e.g., http://localhost:8010)",
    )
    parser.add_argument(
        "--purple-agent-id",
        default="purple-agent",
        help="Identifier for the purple agent being evaluated (default: purple-agent)",
    )
    parser.add_argument(
        "--simulation-date",
        default=None,
        help="Simulation date YYYY-MM-DD (default: previous year start)",
    )
    parser.add_argument(
        "--difficulty",
        action="append",
        choices=[d.value for d in TaskDifficulty],
        help="Optional difficulty filter; repeatable",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of examples to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-debate",
        action="store_true",
        help="Skip debate phase",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file for summary/results",
    )
    return parser.parse_args()


def _load_templates(args: argparse.Namespace):
    provider = CsvFinanceDatasetProvider(args.dataset_path)
    templates = provider.to_templates()
    if args.difficulty:
        difficulties = {TaskDifficulty(d) for d in args.difficulty}
        templates = [t for t in templates if t.difficulty in difficulties]
    if args.limit:
        templates = templates[: args.limit]
    return provider, templates


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    if args.simulation_date:
        sim_date = datetime.strptime(args.simulation_date, "%Y-%m-%d")
    else:
        sim_date = datetime(datetime.now().year - 1, 1, 1)

    provider, templates = _load_templates(args)
    generator = DynamicTaskGenerator(dataset_provider=provider)
    evaluator = ComprehensiveEvaluator()
    agent = PurpleHTTPAgentClient(
        base_url=args.purple_endpoint,
        agent_id=args.purple_agent_id,
        model="purple-http",
    )

    results: List[dict[str, Any]] = []

    async def process_template(tpl):
        """Process a single template and return result dict."""
        task = await generator.generate_task(tpl.template_id, sim_date)
        if not task:
            return {
                "template_id": tpl.template_id,
                "purple_agent_id": args.purple_agent_id,
                "purple_endpoint": args.purple_endpoint,
                "error": "task_generation_failed",
            }

        eval_result = await evaluator.run_full_evaluation(
            task=task,
            agent_client=agent,
            conduct_debate=not args.no_debate,
        )

        alpha = getattr(eval_result.alpha_score, "score", None)
        role = getattr(eval_result.role_score, "total", None)
        debate_obj = getattr(eval_result, "debate_result", None)
        debate = getattr(debate_obj, "debate_multiplier", None) if debate_obj else None
        cost_obj = getattr(eval_result, "cost_breakdown", None)
        cost = getattr(cost_obj, "total_cost_usd", None) if cost_obj else None

        # Extract detailed evaluation components
        role_score_obj = eval_result.role_score
        macro_score = getattr(role_score_obj, "macro_score", None)
        exec_score = getattr(role_score_obj, "execution_score", None)

        # Extract tool usage details
        tool_calls = [
            {"tool": tc.tool_name, "success": tc.success}
            for tc in (eval_result.tool_calls or [])
        ]
        code_executions = [
            {"code_snippet": ce.code[:200] + "..." if len(ce.code) > 200 else ce.code, "success": ce.success}
            for ce in (eval_result.code_executions or [])
        ]

        # Extract debate details
        debate_conviction = getattr(debate_obj, "conviction_level", None) if debate_obj else None
        debate_conviction_str = debate_conviction.value if debate_conviction else None

        # Extract ground truth for reference
        ground_truth = task.ground_truth

        return {
            "task_id": task.question_id,
            "template_id": tpl.template_id,
            "purple_agent_id": args.purple_agent_id,
            "purple_endpoint": args.purple_endpoint,
            "category": task.category.value,
            "difficulty": task.difficulty.value,
            "ticker": task.ticker,
            "fiscal_year": task.fiscal_year,
            "question": task.question,
            "rubric": {
                "criteria": task.rubric.criteria,
                "penalty_conditions": task.rubric.penalty_conditions,
                "max_score": task.rubric.max_score,
            },
            "ground_truth": {
                "macro_thesis": ground_truth.macro_thesis,
                "key_themes": ground_truth.key_themes,
                "expected_recommendation": ground_truth.expected_recommendation,
                "numerical_answer": ground_truth.numerical_answer,
                "tolerance": ground_truth.tolerance,
            },
            "purple_agent_response": eval_result.agent_analysis,
            "debate": {
                "green_agent_counter_argument": debate_obj.counter_argument if debate_obj else None,
                "purple_agent_rebuttal": debate_obj.agent_rebuttal if debate_obj else None,
                "multiplier": debate,
                "conviction_level": debate_conviction_str,
                "new_evidence_provided": debate_obj.new_evidence_provided if debate_obj else None,
                "hallucination_detected": debate_obj.hallucination_detected if debate_obj else None,
                "immediate_concession": debate_obj.immediate_concession if debate_obj else None,
                "debate_feedback": debate_obj.feedback if debate_obj else None,
            },
            "green_agent_evaluation": {
                "execution_feedback": eval_result.execution_feedback,
                "macro_feedback": eval_result.macro_feedback,
            },
            "agent_activity": {
                "tool_calls": tool_calls,
                "code_executions": code_executions,
                "total_execution_time_seconds": eval_result.total_execution_time_seconds,
                "total_llm_calls": eval_result.total_llm_calls,
                "total_tokens": eval_result.total_tokens,
            },
            "scores": {
                "alpha_score": alpha,
                "role_score": role,
                "macro_score": macro_score,
                "execution_score": exec_score,
            },
            "cost": cost,
            "error": None,
        }

    async def run_all():
        """Run evaluation for all templates."""
        for tpl in templates:
            try:
                result = await process_template(tpl)
                results.append(result)
            except Exception as e:
                results.append({
                    "template_id": tpl.template_id,
                    "purple_agent_id": args.purple_agent_id,
                    "purple_endpoint": args.purple_endpoint,
                    "error": str(e),
                })

    asyncio.run(run_all())

    # Aggregate
    alpha_values = [
        r["scores"]["alpha_score"]
        for r in results
        if r.get("scores") and isinstance(r["scores"].get("alpha_score"), (int, float))
    ]
    role_values = [
        r["scores"]["role_score"]
        for r in results
        if r.get("scores") and isinstance(r["scores"].get("role_score"), (int, float))
    ]
    cost_values = [r["cost"] for r in results if isinstance(r.get("cost"), (int, float))]

    summary: dict[str, Any] = {
        "purple_agent_id": args.purple_agent_id,
        "purple_endpoint": args.purple_endpoint,
        "dataset_path": args.dataset_path,
        "simulation_date": sim_date.isoformat(),
        "evaluation_timestamp": datetime.now().isoformat(),
        "count": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "alpha_mean": statistics.mean(alpha_values) if alpha_values else None,
        "alpha_median": statistics.median(alpha_values) if alpha_values else None,
        "alpha_min": min(alpha_values) if alpha_values else None,
        "alpha_max": max(alpha_values) if alpha_values else None,
        "alpha_stdev": statistics.stdev(alpha_values) if len(alpha_values) > 1 else None,
        "role_mean": statistics.mean(role_values) if role_values else None,
        "total_cost": sum(cost_values) if cost_values else None,
        "by_difficulty": {},
        "by_category": {},
    }
    # difficulty counts
    diff_counts: dict[str, int] = {}
    for r in results:
        diff = r.get("difficulty")
        if diff:
            diff_counts[diff] = diff_counts.get(diff, 0) + 1
    summary["by_difficulty"] = diff_counts

    # category counts
    cat_counts: dict[str, int] = {}
    for r in results:
        cat = r.get("category")
        if cat:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    summary["by_category"] = cat_counts

    output = {"summary": summary, "results": results}
    try:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2))
        output_path = args.output
    except PermissionError:
        fallback = "/tmp/summary.json"
        Path(fallback).write_text(json.dumps(output, indent=2))
        output_path = fallback
        print(f"Warning: cannot write to {args.output}, wrote to {fallback} instead")

    # Print brief summary
    print(json.dumps(summary, indent=2))
    print(f"Summary written to: {output_path}")


if __name__ == "__main__":
    main()
