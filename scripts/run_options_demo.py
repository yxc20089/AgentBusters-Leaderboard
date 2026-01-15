#!/usr/bin/env python3
"""
Options Trading Demo: Run Purple Agent options analysis and Green Agent evaluation

This demonstrates the Options Alpha Challenge with:
1. Purple Agent constructing options strategies
2. Green Agent evaluating strategy quality, Greeks, and P&L
3. Adversarial debate on strategy risks

Usage:
    python scripts/run_options_demo.py
    python scripts/run_options_demo.py --task volatility
    python scripts/run_options_demo.py --ticker TSLA
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, "src")

from cio_agent.models import (
    Task,
    TaskCategory,
    TaskDifficulty,
    TaskRubric,
    GroundTruth,
    FinancialData,
    AgentResponse,
)
from cio_agent.evaluator import ComprehensiveEvaluator, EvaluationReporter
from cio_agent.task_generator import DynamicTaskGenerator, FABDataset
from evaluators.options import OptionsEvaluator, OPTIONS_CATEGORIES
from purple_agent.executor import FinanceAgentExecutor

console = Console()


def create_iron_condor_task(ticker: str = "SPY") -> Task:
    """Create an iron condor strategy construction task."""
    return Task(
        question_id=f"OPTIONS_IRON_CONDOR_{ticker}",
        category=TaskCategory.STRATEGY_CONSTRUCTION,
        question=(
            f"Construct an iron condor on {ticker} with 30-day expiration. "
            "Select strikes to achieve at least 70% probability of profit. "
            "Show all Greeks for the position, max profit, max loss, and breakeven points."
        ),
        ticker=ticker,
        fiscal_year=2025,
        simulation_date=datetime(2025, 1, 15),
        ground_truth=GroundTruth(
            macro_thesis=(
                "Iron condor is a neutral strategy that profits from low volatility. "
                "Ideal when IV rank is high (sell premium) and expecting range-bound movement. "
                "Risk is defined and limited to the width of the spread minus credit received."
            ),
            key_themes=[
                "iron condor",
                "delta neutral",
                "theta positive",
                "defined risk",
                "probability of profit",
                "breakeven points",
            ],
            financials=FinancialData(),
        ),
        difficulty=TaskDifficulty.HARD,
        rubric=TaskRubric(
            criteria=[
                "Correctly identify all 4 legs of the iron condor",
                "Calculate and display all Greeks (delta, gamma, theta, vega)",
                "Calculate max profit and max loss",
                "Determine breakeven points",
                "Estimate probability of profit",
            ],
            mandatory_elements=[
                "4 option legs",
                "max profit",
                "max loss",
                "breakeven points",
                "delta",
                "gamma",
                "theta",
                "vega",
            ],
        ),
        requires_code_execution=True,
    )


def create_volatility_task(ticker: str = "TSLA") -> Task:
    """Create a volatility analysis and trading task."""
    return Task(
        question_id=f"OPTIONS_VOLATILITY_{ticker}",
        category=TaskCategory.VOLATILITY_TRADING,
        question=(
            f"Analyze {ticker}'s current implied volatility. Is IV overpriced or "
            "underpriced relative to historical volatility? Design a trade to exploit "
            "any mispricing. Include IV rank, IV percentile, and recommended strategy."
        ),
        ticker=ticker,
        fiscal_year=2025,
        simulation_date=datetime(2025, 1, 15),
        ground_truth=GroundTruth(
            macro_thesis=(
                "Volatility trading exploits the difference between implied and realized volatility. "
                "When IV is high relative to historical (IV rank > 50%), sell premium strategies work well. "
                "When IV is low, buying options or long volatility strategies may be preferred."
            ),
            key_themes=[
                "implied volatility",
                "historical volatility",
                "IV rank",
                "IV percentile",
                "volatility premium",
                "mean reversion",
            ],
            financials=FinancialData(),
        ),
        difficulty=TaskDifficulty.HARD,
        rubric=TaskRubric(
            criteria=[
                "Calculate IV rank and IV percentile",
                "Compare IV to historical volatility",
                "Identify if IV is overpriced or underpriced",
                "Recommend appropriate strategy",
                "Explain risk/reward of the trade",
            ],
            mandatory_elements=[
                "IV analysis",
                "HV comparison",
                "trade recommendation",
            ],
        ),
        requires_code_execution=True,
    )


def create_greeks_hedge_task(ticker: str = "AAPL") -> Task:
    """Create a Greeks analysis and hedging task."""
    return Task(
        question_id=f"OPTIONS_GREEKS_HEDGE_{ticker}",
        category=TaskCategory.GREEKS_ANALYSIS,
        question=(
            f"Your portfolio has -500 delta exposure from {ticker} options. "
            "Design a hedge using options or stock to neutralize the delta. "
            "Show the resulting portfolio Greeks after hedging."
        ),
        ticker=ticker,
        fiscal_year=2025,
        simulation_date=datetime(2025, 1, 15),
        ground_truth=GroundTruth(
            macro_thesis=(
                "Delta hedging neutralizes directional exposure. Can hedge with stock (delta = 1.0) "
                "or options. Options provide additional Greek exposures. Consider cost and "
                "ongoing rebalancing requirements. Gamma determines how often rehedging is needed."
            ),
            key_themes=[
                "delta hedging",
                "portfolio Greeks",
                "delta neutral",
                "gamma exposure",
                "rebalancing",
            ],
            financials=FinancialData(),
        ),
        difficulty=TaskDifficulty.MEDIUM,
        rubric=TaskRubric(
            criteria=[
                "Understand current delta exposure",
                "Propose valid hedge",
                "Calculate hedge size correctly",
                "Show resulting portfolio Greeks",
            ],
            mandatory_elements=[
                "hedge instrument",
                "quantity",
                "resulting portfolio delta",
            ],
        ),
        requires_code_execution=True,
    )


def create_risk_management_task(ticker: str = "SPY") -> Task:
    """Create a risk management and position sizing task."""
    return Task(
        question_id=f"OPTIONS_RISK_MGMT_{ticker}",
        category=TaskCategory.RISK_MANAGEMENT,
        question=(
            f"Size a {ticker} strangle position to stay within a $10,000 VaR limit "
            "at 95% confidence over 1 day. Show your VaR calculation methodology "
            "and the resulting position size."
        ),
        ticker=ticker,
        fiscal_year=2025,
        simulation_date=datetime(2025, 1, 15),
        ground_truth=GroundTruth(
            macro_thesis=(
                "VaR-based position sizing ensures risk remains within tolerance. "
                "Calculate position VaR using historical, parametric, or Monte Carlo methods. "
                "Strangle has unlimited risk on both sides, so sizing is critical."
            ),
            key_themes=[
                "Value at Risk",
                "position sizing",
                "risk limits",
                "strangle risk",
                "confidence level",
            ],
            financials=FinancialData(),
        ),
        difficulty=TaskDifficulty.EXPERT,
        rubric=TaskRubric(
            criteria=[
                "Calculate position VaR correctly",
                "Size position to meet VaR constraint",
                "Explain VaR methodology",
                "Consider tail risks",
            ],
            mandatory_elements=[
                "position size",
                "VaR calculation",
                "risk parameters",
            ],
        ),
        requires_code_execution=True,
    )


TASK_CREATORS = {
    "iron_condor": create_iron_condor_task,
    "volatility": create_volatility_task,
    "greeks": create_greeks_hedge_task,
    "risk": create_risk_management_task,
}


async def run_purple_agent_analysis(task: Task) -> AgentResponse:
    """Run Purple Agent to analyze an options task."""
    executor = FinanceAgentExecutor(
        llm_client=None,  # Use fallback response
        simulation_date=task.simulation_date,
    )

    # Parse and gather data
    task_info = executor._parse_task(task.question)
    task_info["tickers"] = [task.ticker]
    financial_data = await executor._gather_data(task_info)

    # Generate analysis
    analysis = await executor._generate_analysis(
        user_input=task.question,
        task_info=task_info,
        financial_data=financial_data,
    )

    return AgentResponse(
        agent_id="purple-options-agent",
        task_id=task.question_id,
        analysis=analysis,
        recommendation=f"Execute {task_info['task_type']} strategy on {task.ticker}",
        extracted_financials=FinancialData(),
    )


async def run_options_demo(task_type: str = "iron_condor", ticker: str = "SPY"):
    """Run the full options demo pipeline."""
    console.print(Panel.fit(
        "[bold blue]AgentBusters Options Alpha Challenge Demo[/bold blue]\n\n"
        "Purple Agent (Options Trader) vs Green Agent (Evaluator)\n"
        f"Task Type: {task_type}\n"
        f"Ticker: {ticker}",
        title="[bold]Options Trading Evaluation[/bold]"
    ))

    # Create the evaluation task
    console.print("\n[cyan]1. Creating options evaluation task...[/cyan]")

    if task_type in TASK_CREATORS:
        task = TASK_CREATORS[task_type](ticker)
    else:
        # Use dynamic task generator
        dataset = FABDataset.load_sample_questions()
        gen = DynamicTaskGenerator(fab_dataset=dataset)
        task = await gen.generate_task(
            template_id=f"OPT_020",  # Iron condor
            new_ticker=ticker,
            simulation_date=datetime(2025, 1, 15),
        )

    console.print(f"   Category: {task.category.value}")
    console.print(f"   Ticker: {task.ticker}")
    console.print(f"   Difficulty: {task.difficulty.value}")
    console.print(f"   Question: {task.question[:100]}...")

    # Run Purple Agent
    console.print("\n[cyan]2. Running Purple Agent options analysis...[/cyan]")
    response = await run_purple_agent_analysis(task)
    console.print(f"   Analysis length: {len(response.analysis)} chars")

    # Show sample of analysis
    console.print("\n[dim]Purple Agent Analysis (excerpt):[/dim]")
    console.print(Panel(response.analysis[:500] + "...", title="Analysis"))

    # Run Green Agent evaluation
    console.print("\n[cyan]3. Running Green Agent evaluation...[/cyan]")

    # Check if this is an options task
    is_options = task.category in OPTIONS_CATEGORIES
    console.print(f"   Options task detected: {is_options}")

    # Run options-specific evaluation
    options_evaluator = OptionsEvaluator(task=task)
    options_result = await options_evaluator.score(response)

    # Display options-specific scores
    scores_table = Table(title="Options Evaluation Scores")
    scores_table.add_column("Dimension", style="cyan")
    scores_table.add_column("Score", justify="right")
    scores_table.add_column("Weight", justify="right")

    scores_table.add_row("P&L Accuracy", f"{options_result.pnl_accuracy:.1f}/100", "25%")
    scores_table.add_row("Greeks Accuracy", f"{options_result.greeks_accuracy:.1f}/100", "25%")
    scores_table.add_row("Strategy Quality", f"{options_result.strategy_quality:.1f}/100", "25%")
    scores_table.add_row("Risk Management", f"{options_result.risk_management:.1f}/100", "25%")
    scores_table.add_row("[bold]Final Score[/bold]", f"[bold]{options_result.score:.1f}/100[/bold]", "[bold]100%[/bold]")

    console.print(scores_table)

    # Display feedback
    console.print(f"\n[cyan]Feedback:[/cyan]")
    console.print(Panel(options_result.feedback, title="Evaluation Feedback"))

    # Now run comprehensive evaluation
    console.print("\n[cyan]4. Running comprehensive evaluation (with debate)...[/cyan]")
    evaluator = ComprehensiveEvaluator()
    result = await evaluator.evaluate_response(
        task=task,
        agent_response=response,
    )

    # Display final results
    console.print("\n" + "=" * 60)
    console.print("[bold green]EVALUATION COMPLETE[/bold green]")
    console.print("=" * 60)

    summary = EvaluationReporter.generate_summary(result)
    console.print(summary)

    # Final score panel
    console.print(Panel.fit(
        f"[bold green]Final Alpha Score: {result.alpha_score.score:.2f}[/bold green]\n\n"
        f"Options Score: {options_result.score:.1f}/100\n"
        f"Role Score: {result.role_score.total:.1f}/100\n"
        f"Debate Multiplier: {result.debate_result.debate_multiplier}x",
        title="[bold]Final Result[/bold]"
    ))

    return result, options_result


async def run_all_options_tasks(ticker: str = "SPY"):
    """Run all options task types for a comprehensive demo."""
    console.print(Panel.fit(
        "[bold blue]Running All Options Task Types[/bold blue]\n\n"
        f"Ticker: {ticker}\n"
        "Tasks: Iron Condor, Volatility, Greeks, Risk Management"
    ))

    results = []
    for task_type in TASK_CREATORS.keys():
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold]Task: {task_type.upper()}[/bold]")
        console.print("=" * 60)

        result, options_result = await run_options_demo(task_type, ticker)
        results.append({
            "task_type": task_type,
            "options_score": options_result.score,
            "alpha_score": result.alpha_score.score,
        })

    # Summary table
    console.print("\n" + "=" * 60)
    console.print("[bold]SUMMARY OF ALL TASKS[/bold]")
    console.print("=" * 60)

    summary_table = Table(title="Options Evaluation Summary")
    summary_table.add_column("Task Type", style="cyan")
    summary_table.add_column("Options Score", justify="right")
    summary_table.add_column("Alpha Score", justify="right")

    for r in results:
        summary_table.add_row(
            r["task_type"],
            f"{r['options_score']:.1f}",
            f"{r['alpha_score']:.2f}",
        )

    console.print(summary_table)


def main():
    parser = argparse.ArgumentParser(description="Options Trading Demo")
    parser.add_argument(
        "--task",
        choices=list(TASK_CREATORS.keys()) + ["all"],
        default="iron_condor",
        help="Task type to run",
    )
    parser.add_argument(
        "--ticker",
        default="SPY",
        help="Stock ticker symbol",
    )

    args = parser.parse_args()

    if args.task == "all":
        asyncio.run(run_all_options_tasks(args.ticker))
    else:
        asyncio.run(run_options_demo(args.task, args.ticker))


if __name__ == "__main__":
    main()
