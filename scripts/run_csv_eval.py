"""
Batch evaluation runner for CSV datasets.

Usage:
    python -m scripts.run_csv_eval \
      --dataset-path /app/data/public.csv \
      --simulation-date 2024-12-31 \
      --difficulty medium \
      --output /data/results/summary.json

Reads a CSV via CsvFinanceDatasetProvider, generates tasks with DynamicTaskGenerator,
runs ComprehensiveEvaluator with a MockAgentClient, and writes per-task results plus
aggregated summary to the output JSON file.
"""

import argparse
import asyncio
import json
import random
import statistics
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, List

from cio_agent.datasets.csv_provider import CsvFinanceDatasetProvider
from cio_agent.evaluator import ComprehensiveEvaluator
from cio_agent.models import TaskDifficulty
from cio_agent.orchestrator import MockAgentClient
from cio_agent.task_generator import DynamicTaskGenerator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate CSV finance tasks.")
    parser.add_argument("--dataset-path", required=True, help="Path to CSV dataset")
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
    agent = MockAgentClient(agent_id="batch-agent", model="gpt-4o")

    results: List[dict[str, Any]] = []

    async def process_template(tpl):
        """Process a single template and return result dict."""
        task = await generator.generate_task(tpl.template_id, sim_date)
        if not task:
            return {
                "template_id": tpl.template_id,
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

        return {
            "task_id": task.question_id,
            "template_id": tpl.template_id,
            "category": task.category.value,
            "difficulty": task.difficulty.value,
            "alpha_score": alpha,
            "role_score": role,
            "debate_multiplier": debate,
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
                    "error": str(e),
                })

    asyncio.run(run_all())

    # Aggregate
    alpha_values = [r["alpha_score"] for r in results if isinstance(r.get("alpha_score"), (int, float))]
    summary: dict[str, Any] = {
        "count": len(results),
        "alpha_mean": statistics.mean(alpha_values) if alpha_values else None,
        "alpha_median": statistics.median(alpha_values) if alpha_values else None,
        "alpha_min": min(alpha_values) if alpha_values else None,
        "alpha_max": max(alpha_values) if alpha_values else None,
        "by_difficulty": {},
    }
    # difficulty counts
    diff_counts: dict[str, int] = {}
    for r in results:
        diff = r.get("difficulty")
        if diff:
            diff_counts[diff] = diff_counts.get(diff, 0) + 1
    summary["by_difficulty"] = diff_counts

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
