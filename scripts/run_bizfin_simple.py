"""
Simple evaluation runner for BizFinBench.v2 using dataset-specific evaluator.

This script uses BizFinBenchEvaluator for exact-match evaluation instead of
the LLM-based ComprehensiveEvaluator.

Usage:
    python -m scripts.run_bizfin_simple \
        --dataset-path data/BizFinBench.v2 \
        --task-type event_logic_reasoning \
        --language en \
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

from cio_agent.datasets.bizfinbench_provider import BizFinBenchProvider
from evaluators.bizfinbench_evaluator import BizFinBenchEvaluator
from evaluators.llm_utils import should_use_llm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple BizFinBench evaluation.")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to BizFinBench.v2 directory",
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=BizFinBenchProvider.list_task_types(),
        help="BizFinBench task type to evaluate",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "cn"],
        help="Language: 'en' or 'cn'",
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
        help="Enable LLM grading for BizFinBench evaluation",
    )
    eval_group.add_argument(
        "--no-eval-llm",
        action="store_true",
        help="Disable LLM grading for BizFinBench evaluation",
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
    provider = BizFinBenchProvider(
        base_path=args.dataset_path,
        task_type=args.task_type,
        language=args.language,
        limit=args.limit,
    )
    examples = provider.load()
    
    print(f"Loaded {len(examples)} examples from BizFinBench.v2")
    print(f"Task type: {args.task_type}, Language: {args.language}")
    print(f"Purple endpoint: {args.purple_endpoint}")
    print("-" * 60)
    
    # Initialize evaluator
    if args.eval_llm:
        use_llm = True
    elif args.no_eval_llm:
        use_llm = False
    else:
        use_llm = should_use_llm()

    evaluator = BizFinBenchEvaluator(
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
            
            # Evaluate
            eval_result = evaluator.evaluate(
                predicted=predicted,
                expected=example.answer,
                task_type=args.task_type,
                question=example.question,
            )
            
            results.append({
                "example_id": example.example_id,
                "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                "expected": example.answer[:100] + "..." if len(example.answer) > 100 else example.answer,
                "predicted": predicted[:200] + "..." if len(predicted) > 200 else predicted,
                "score": eval_result.score,
                "is_correct": eval_result.is_correct,
                "feedback": eval_result.feedback,
                "details": eval_result.details,
            })
            
            status = "✓" if eval_result.is_correct else "✗"
            print(f"  {status} Score: {eval_result.score:.2f}")
    
    # Aggregate results
    scores = [r["score"] for r in results]
    correct_count = sum(1 for r in results if r["is_correct"])
    
    summary = {
        "dataset_path": args.dataset_path,
        "task_type": args.task_type,
        "language": args.language,
        "purple_endpoint": args.purple_endpoint,
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_examples": len(results),
        "correct_count": correct_count,
        "accuracy": correct_count / len(results) if results else 0.0,
        "mean_score": statistics.mean(scores) if scores else 0.0,
        "score_stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
    }
    
    output = {"summary": summary, "results": results}
    
    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total examples: {summary['total_examples']}")
    print(f"Correct: {summary['correct_count']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Mean score: {summary['mean_score']:.4f}")
    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
