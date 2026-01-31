#!/usr/bin/env python
"""
Adversarial Robustness Evaluation Script.

Tests Purple Agent robustness using the new Adversarial Robustness System.
Generates perturbations and measures consistency across attacks.

Usage:
    # Basic robustness test (5 questions)
    python scripts/run_robustness_eval.py --purple-url http://localhost:9110 --num-questions 5

    # Full robustness evaluation with specific attacks
    python scripts/run_robustness_eval.py \
        --purple-url http://localhost:9110 \
        --attacks paraphrase typo distraction \
        --num-questions 20 \
        --output results/robustness_report.json

    # Use custom dataset
    python scripts/run_robustness_eval.py \
        --purple-url http://localhost:9110 \
        --dataset finance-agent/data/public.csv \
        --num-questions 10
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from cio_agent.adversarial_robustness import (
    AdversarialPerturbationEngine,
    RobustnessEvaluator,
    RobustnessResult,
    AttackType,
)

console = Console()


async def call_purple_agent(
    client: httpx.AsyncClient,
    purple_url: str,
    question: str,
    timeout: float = 60.0,
) -> str:
    """Send question to Purple Agent and get answer."""
    # Try A2A format first
    a2a_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": question}],
            }
        },
    }
    
    try:
        response = await client.post(
            f"{purple_url}/",
            json=a2a_payload,
            timeout=timeout,
        )
        
        if response.status_code == 200:
            result = response.json()
            if "result" in result:
                parts = result.get("result", {}).get("message", {}).get("parts", [])
                for part in parts:
                    if part.get("kind") == "text":
                        return part.get("text", "")
        
        # Fallback: try simple endpoint
        simple_response = await client.post(
            f"{purple_url}/analyze",
            json={"question": question},
            timeout=timeout,
        )
        if simple_response.status_code == 200:
            return simple_response.json().get("answer", "")
            
    except Exception as e:
        console.print(f"[red]Error calling Purple Agent: {e}[/red]")
        return ""
    
    return ""


def load_questions_from_csv(csv_path: str, limit: int = 10) -> list[dict]:
    """Load questions from CSV file."""
    import csv
    
    questions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            questions.append({
                "question": row.get("Question", ""),
                "expected_answer": row.get("Answer", ""),
                "question_type": row.get("Question Type", ""),
            })
    
    return questions


def display_results(report: dict[str, Any]):
    """Display robustness report in rich format."""
    # Header panel
    grade = report.get("overall_grade", "F")
    score = report.get("overall_robustness_score", 0.0)
    
    grade_colors = {"A": "green", "B": "blue", "C": "yellow", "D": "orange", "F": "red"}
    color = grade_colors.get(grade, "white")
    
    console.print(Panel(
        f"[bold {color}]Robustness Grade: {grade}[/bold {color}]\n"
        f"Overall Score: {score:.1%}",
        title="🛡️ Adversarial Robustness Report",
        expand=False,
    ))
    
    # Attack type breakdown
    attack_scores = report.get("attack_type_scores", {})
    if attack_scores:
        table = Table(title="Attack Type Scores")
        table.add_column("Attack Type", style="cyan")
        table.add_column("Score", style="magenta")
        table.add_column("Status", style="white")
        
        for attack_type, score in sorted(attack_scores.items(), key=lambda x: x[1]):
            if score >= 0.8:
                status = "[green]✓ Robust[/green]"
            elif score >= 0.6:
                status = "[yellow]⚠ Moderate[/yellow]"
            else:
                status = "[red]✗ Vulnerable[/red]"
            
            table.add_row(attack_type, f"{score:.1%}", status)
        
        console.print(table)
    
    # Vulnerabilities
    vulnerabilities = report.get("sample_vulnerabilities", [])
    if vulnerabilities:
        console.print("\n[bold red]⚠️ Vulnerabilities Found:[/bold red]")
        for i, vuln in enumerate(vulnerabilities[:5], 1):
            console.print(f"  {i}. [{vuln.get('attack_type')}] {vuln.get('issue')}")
    
    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        console.print("\n[bold cyan]📋 Recommendations:[/bold cyan]")
        for rec in recommendations:
            console.print(f"  • {rec}")
    
    # Summary stats
    console.print(f"\n[dim]Total questions tested: {report.get('total_questions', 0)}[/dim]")
    console.print(f"[dim]Total vulnerabilities: {report.get('total_vulnerabilities', 0)}[/dim]")


async def main():
    parser = argparse.ArgumentParser(
        description="Test Purple Agent robustness against adversarial perturbations"
    )
    parser.add_argument(
        "--purple-url",
        default="http://localhost:9110",
        help="Purple Agent URL",
    )
    parser.add_argument(
        "--dataset",
        default="finance-agent/data/public.csv",
        help="Path to CSV dataset with questions",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of questions to test",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=[a.value for a in AttackType],
        default=["paraphrase", "typo", "distraction"],
        help="Attack types to test",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.5,
        help="Attack intensity (0.0-1.0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for JSON report",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    
    args = parser.parse_args()
    
    console.print("[bold cyan]🛡️ AgentBusters Adversarial Robustness Evaluation[/bold cyan]\n")
    
    # Load questions
    if not Path(args.dataset).exists():
        console.print(f"[red]Dataset not found: {args.dataset}[/red]")
        return 1
    
    questions = load_questions_from_csv(args.dataset, args.num_questions)
    console.print(f"Loaded {len(questions)} questions from {args.dataset}")
    
    # Parse attack types
    attack_types = [AttackType(a) for a in args.attacks]
    console.print(f"Attack types: {', '.join(a.value for a in attack_types)}")
    
    # Initialize evaluator
    engine = AdversarialPerturbationEngine(
        seed=42,
        attack_intensity=args.intensity,
    )
    evaluator = RobustnessEvaluator(perturbation_engine=engine)
    
    # Create answer function
    async with httpx.AsyncClient() as client:
        async def answer_func(question: str) -> str:
            return await call_purple_agent(client, args.purple_url, question)
        
        # Run robustness tests
        results: list[RobustnessResult] = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Testing robustness...",
                total=len(questions),
            )
            
            for i, q in enumerate(questions):
                progress.update(
                    task,
                    description=f"Testing Q{i+1}/{len(questions)}: {q['question'][:50]}..."
                )
                
                try:
                    result = await evaluator.evaluate_robustness(
                        question=q["question"],
                        answer_func=answer_func,
                        expected_answer=q.get("expected_answer"),
                        attack_types=attack_types,
                    )
                    results.append(result)
                    
                    if args.verbose:
                        console.print(
                            f"  Q{i+1}: Grade={result.robustness_grade}, "
                            f"Consistency={result.consistency_score:.1%}"
                        )
                        
                except Exception as e:
                    console.print(f"[red]Error on Q{i+1}: {e}[/red]")
                
                progress.advance(task)
    
    # Generate report
    report = evaluator.generate_robustness_report(results)
    
    # Display results
    console.print()
    display_results(report)
    
    # Save to file if requested
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            # Convert AttackType keys to strings for JSON
            report_serializable = report.copy()
            report_serializable["attack_type_scores"] = {
                str(k): v for k, v in report.get("attack_type_scores", {}).items()
            }
            json.dump(report_serializable, f, indent=2, default=str)
        console.print(f"\n[green]Report saved to {args.output}[/green]")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
