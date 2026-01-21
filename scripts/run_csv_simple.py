"""
Simple evaluation runner for public.csv using dataset-specific evaluator.

This script uses PublicCsvEvaluator for rubric-based evaluation instead of
the LLM-based ComprehensiveEvaluator.

Usage:
    python -m scripts.run_csv_simple \
        --dataset-path finance-agent/data/public.csv \
        --purple-endpoint http://localhost:9110 \
        --output /tmp/results.json \
        --limit 5 \
        --eval-llm
"""

import argparse
import asyncio
import json
import httpx
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from cio_agent.datasets.csv_provider import CsvFinanceDatasetProvider
from evaluators.public_csv_evaluator import PublicCsvEvaluator
from evaluators.llm_utils import should_use_llm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple public.csv evaluation.")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to public.csv file",
    )
    parser.add_argument(
        "--purple-endpoint",
        required=True,
        help="Purple Agent endpoint URL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds",
    )
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--eval-llm",
        action="store_true",
        help="Enable LLM grading for rubric evaluation",
    )
    eval_group.add_argument(
        "--no-eval-llm",
        action="store_true",
        help="Disable LLM grading for rubric evaluation",
    )
    parser.add_argument(
        "--eval-llm-model",
        type=str,
        help="Model override for LLM grading (e.g., gpt-4o-mini)",
    )
    parser.add_argument(
        "--eval-llm-temperature",
        type=float,
        help="Temperature for LLM grading (default 0.0)",
    )
    return parser.parse_args()


async def call_purple_agent(
    client: httpx.AsyncClient,
    endpoint: str,
    question: str,
    timeout: float
) -> str:
    """Call Purple Agent to get response."""
    try:
        response = await client.post(
            f"{endpoint}/analyze",
            json={"question": question},
            timeout=timeout,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("analysis", data.get("answer", str(data)))
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"Error: {e}"


async def main() -> None:
    args = _parse_args()
    
    # Load dataset
    provider = CsvFinanceDatasetProvider(path=args.dataset_path)
    examples = provider.load()
    
    if args.limit:
        examples = examples[:args.limit]
    
    print(f"Loaded {len(examples)} examples from public.csv")
    print(f"Purple endpoint: {args.purple_endpoint}")
    print("-" * 60)
    
    # Initialize evaluator
    if args.eval_llm:
        use_llm = True
    elif args.no_eval_llm:
        use_llm = False
    else:
        use_llm = should_use_llm()

    evaluator = PublicCsvEvaluator(
        use_llm=use_llm,
        llm_model=args.eval_llm_model,
        llm_temperature=args.eval_llm_temperature,
    )
    
    results: List[Dict[str, Any]] = []
    
    async with httpx.AsyncClient() as client:
        for idx, example in enumerate(examples):
            print(f"[{idx+1}/{len(examples)}] {example.example_id}")
            
            # Get response from Purple Agent
            predicted = await call_purple_agent(
                client,
                args.purple_endpoint,
                example.question,
                args.timeout,
            )
            
            # Extract rubric (stored in example.rubric)
            rubric_list = []
            if example.rubric:
                # Convert TaskRubric to list of dicts for evaluator
                for criterion in example.rubric.criteria:
                    rubric_list.append({"operator": "correctness", "criteria": criterion})
                for penalty in example.rubric.penalty_conditions:
                    rubric_list.append({"operator": "contradiction", "criteria": penalty})
            
            # Evaluate
            eval_result = evaluator.evaluate(
                predicted=predicted,
                expected=example.answer,
                rubric=rubric_list if rubric_list else None,
                question=example.question,
            )
            
            results.append({
                "example_id": example.example_id,
                "category": example.category.value,
                "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                "expected": example.answer[:100] + "..." if len(example.answer) > 100 else example.answer,
                "predicted": predicted[:200] + "..." if len(predicted) > 200 else predicted,
                "score": eval_result.score,
                "correct_count": eval_result.correct_count,
                "total_count": eval_result.total_count,
                "feedback": eval_result.feedback,
            })
            
            print(f"  Score: {eval_result.score:.2f} ({eval_result.feedback})")
    
    # Aggregate results
    scores = [r["score"] for r in results]
    
    summary = {
        "dataset_path": args.dataset_path,
        "purple_endpoint": args.purple_endpoint,
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_examples": len(results),
        "mean_score": statistics.mean(scores) if scores else 0.0,
        "score_stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
    }
    
    # Category breakdown
    by_category: Dict[str, List[float]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r["score"])
    
    summary["by_category"] = {
        cat: {"count": len(scores), "mean_score": statistics.mean(scores)}
        for cat, scores in by_category.items()
    }
    
    output = {"summary": summary, "results": results}
    
    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total examples: {summary['total_examples']}")
    print(f"Mean score: {summary['mean_score']:.4f}")
    print(f"Score range: {summary['min_score']:.2f} - {summary['max_score']:.2f}")
    print(f"\nBy category:")
    for cat, stats in summary["by_category"].items():
        print(f"  {cat}: {stats['count']} examples, mean={stats['mean_score']:.2f}")
    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
