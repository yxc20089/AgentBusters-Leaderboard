#!/usr/bin/env python3
"""
Test script: NVIDIA Q3 FY2026 Earnings Evaluation

Demonstrates the CIO-Agent FAB++ system by running an evaluation
on NVIDIA's latest earnings (Q3 FY2026, quarter ended October 26, 2025).

Real data sources:
- https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2026
- https://www.cnbc.com/2025/11/19/nvidia-nvda-earnings-report-q3-2026.html
"""

import asyncio
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

# Add src to path
import sys
sys.path.insert(0, "src")

from cio_agent.models import (
    Task,
    TaskCategory,
    TaskDifficulty,
    TaskRubric,
    GroundTruth,
    FinancialData,
)
from cio_agent.orchestrator import MockAgentClient
from cio_agent.evaluator import ComprehensiveEvaluator, EvaluationReporter

console = Console()


async def run_nvidia_evaluation():
    """Run an evaluation on NVIDIA Q3 FY2026 earnings task."""

    console.print(Panel.fit(
        "[bold blue]CIO-Agent FAB++ Test[/bold blue]\n"
        "Task: NVIDIA Q3 FY2026 Earnings Analysis\n"
        "Quarter: Ended October 26, 2025\n"
        "Simulation Date: 2025-11-20 (after Q3 FY26 results)"
    ))

    # Create NVIDIA Q3 FY2026 earnings task with REAL data
    # NVIDIA's fiscal year ends in January, so Q3 FY2026 = Aug-Oct 2025
    # Results announced November 19, 2025
    task = Task(
        question_id="NVIDIA_Q3_FY2026_earnings",
        category=TaskCategory.BEAT_OR_MISS,
        question=(
            "Did NVIDIA beat or miss analyst expectations in Q3 FY2026 "
            "(quarter ended October 26, 2025)? Analyze the earnings results "
            "including revenue, EPS, data center performance, and Blackwell GPU demand."
        ),
        ticker="NVDA",
        fiscal_year=2026,
        simulation_date=datetime(2025, 11, 20),  # After Q3 FY26 results
        ground_truth=GroundTruth(
            macro_thesis=(
                "NVIDIA's Q3 FY2026 results demonstrate unprecedented AI compute demand. "
                "Revenue hit a record $57B (+62% YoY), significantly beating the $54.92B "
                "consensus. EPS of $1.30 beat the $1.25 estimate. Blackwell GPU sales were "
                "'off the charts' with cloud GPUs sold out. Data center revenue of $51.2B "
                "(+66% YoY) drove growth. Q4 guidance of $65B exceeded $61.66B consensus."
            ),
            key_themes=[
                "AI compute demand",
                "Blackwell GPU",
                "data center growth",
                "beat expectations",
                "cloud GPU sold out",
                "record revenue",
            ],
            financials=FinancialData(
                revenue=57_000_000_000,       # $57.0B record revenue
                net_income=31_910_000_000,    # $31.91B net income (+65% YoY)
                gross_margin=0.734,           # 73.4% GAAP gross margin
                eps=1.30,                     # $1.30 GAAP diluted EPS
                extra_fields={
                    "data_center_revenue": 51_200_000_000,  # $51.2B (+66% YoY)
                    "gaming_revenue": 4_300_000_000,         # $4.3B (+30% YoY)
                    "professional_viz_revenue": 760_000_000, # $760M (+56% YoY)
                    "automotive_revenue": 592_000_000,       # $592M (+32% YoY)
                    "yoy_revenue_growth": 0.62,              # 62% YoY
                    "qoq_revenue_growth": 0.22,              # 22% QoQ
                    "consensus_revenue": 54_920_000_000,     # Expected $54.92B
                    "consensus_eps": 1.25,                   # Expected $1.25
                    "q4_guidance": 65_000_000_000,           # Q4 FY26 guidance $65B
                }
            ),
            expected_recommendation="Beat",
            numerical_answer=57_000_000_000,  # Q3 revenue
        ),
        difficulty=TaskDifficulty.MEDIUM,
        rubric=TaskRubric(
            criteria=[
                "Correctly identify beat/miss status",
                "Provide actual vs expected figures (Revenue and EPS)",
                "Analyze data center segment performance",
                "Discuss Blackwell GPU demand and AI compute trends",
                "Mention Q4 guidance",
            ],
            mandatory_elements=[
                "beat or miss determination",
                "revenue figures",
                "EPS figures",
            ],
        ),
        requires_code_execution=False,
    )

    console.print("\n[cyan]Task Details:[/cyan]")
    console.print(f"  Category: {task.category.value}")
    console.print(f"  Ticker: {task.ticker}")
    console.print(f"  Question: {task.question[:100]}...")
    console.print()

    # Create mock agent and evaluator
    agent = MockAgentClient(
        agent_id="nvidia-analyst",
        model="gpt-4o",
    )

    evaluator = ComprehensiveEvaluator()

    # Run evaluation
    console.print("[yellow]Running evaluation...[/yellow]\n")

    result = await evaluator.run_full_evaluation(
        task=task,
        agent_client=agent,
        conduct_debate=True,
    )

    # Get agent response for report
    agent_response = await agent.process_task(task)

    # Print results
    console.print("\n" + "="*60)
    console.print("[bold green]EVALUATION COMPLETE[/bold green]")
    console.print("="*60 + "\n")

    # Print summary
    summary = EvaluationReporter.generate_summary(result)
    console.print(summary)

    # Print detailed scores
    console.print("\n[bold cyan]Detailed Score Breakdown:[/bold cyan]")
    console.print(f"  Macro Score: {result.role_score.macro.score:.1f}/100")
    console.print(f"    - {result.role_score.macro.feedback}")
    console.print(f"  Fundamental Score: {result.role_score.fundamental.score:.1f}/100")
    console.print(f"    - {result.role_score.fundamental.feedback}")
    console.print(f"  Execution Score: {result.role_score.execution.score:.1f}/100")
    console.print(f"    - {result.role_score.execution.feedback}")

    console.print(f"\n[bold cyan]Debate Result:[/bold cyan]")
    console.print(f"  Multiplier: {result.debate_result.debate_multiplier}x")
    console.print(f"  Conviction: {result.debate_result.conviction_level.value}")
    console.print(f"  {result.debate_result.feedback}")

    console.print(f"\n[bold cyan]Efficiency:[/bold cyan]")
    console.print(f"  Total Cost: ${result.cost_breakdown.total_cost_usd:.4f}")
    console.print(f"  Tool Calls: {len(result.tool_calls)}")
    console.print(f"  Temporal Violations: {len(result.lookahead_penalty.violations)}")

    # Final score
    console.print(Panel.fit(
        f"[bold green]Final Alpha Score: {result.alpha_score.score:.2f}[/bold green]\n\n"
        f"Formula: ({result.alpha_score.role_score:.1f} × {result.alpha_score.debate_multiplier}) / "
        f"(ln(1 + {result.alpha_score.cost_usd:.4f}) × (1 + {result.alpha_score.lookahead_penalty}))",
        title="[bold]Result[/bold]"
    ))

    return result


if __name__ == "__main__":
    asyncio.run(run_nvidia_evaluation())
