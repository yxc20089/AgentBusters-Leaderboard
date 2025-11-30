"""
CLI for CIO-Agent FAB++ Evaluator.

Provides command-line interface for:
- Running evaluations
- Generating tasks
- Viewing reports
"""

import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from cio_agent.task_generator import DynamicTaskGenerator, FABDataset
from cio_agent.orchestrator import MockAgentClient
from cio_agent.evaluator import ComprehensiveEvaluator, EvaluationReporter
from cio_agent.models import TaskCategory, TaskDifficulty
from cio_agent.datasets.csv_provider import CsvFinanceDatasetProvider
from cio_agent.a2a_client import PurpleHTTPAgentClient

app = typer.Typer(
    name="cio-agent",
    help="CIO-Agent FAB++ Evaluator - Dynamic Finance Agent Benchmark",
)
console = Console()


@app.command()
def evaluate(
    task_id: str = typer.Option(
        "FAB_001",
        "--task-id", "-t",
        help="FAB question template ID to evaluate"
    ),
    simulation_date: str = typer.Option(
        None,
        "--date", "-d",
        help="Simulation date (YYYY-MM-DD). Defaults to 1 year ago."
    ),
    agent_model: str = typer.Option(
        "gpt-4o",
        "--model", "-m",
        help="Agent model to simulate"
    ),
    purple_base_url: Optional[str] = typer.Option(
        None,
        "--purple-endpoint",
        help="Purple agent base URL (e.g., http://purple-agent:8001 or http://localhost:8001)",
        envvar="PURPLE_ENDPOINT",
    ),
    no_debate: bool = typer.Option(
        False,
        "--no-debate",
        help="Skip adversarial debate phase"
    ),
    output_format: str = typer.Option(
        "summary",
        "--output", "-o",
        help="Output format: summary, markdown, json"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--file", "-f",
        help="Output file path"
    ),
    dataset_path: Optional[Path] = typer.Option(
        None,
        "--dataset-path",
        help="Path to external CSV dataset (e.g., data/public.csv)"
    ),
    difficulty: Optional[list[TaskDifficulty]] = typer.Option(
        None,
        "--difficulty",
        help="Optional difficulty filter when selecting a random task",
    ),
):
    """
    Run a single evaluation on a FAB task.
    """
    # Parse simulation date
    if simulation_date:
        sim_date = datetime.strptime(simulation_date, "%Y-%m-%d")
    else:
        sim_date = datetime(datetime.now().year - 1, 1, 1)

    console.print(Panel.fit(
        f"[bold blue]CIO-Agent FAB++ Evaluator[/bold blue]\n"
        f"Task: {task_id}\n"
        f"Simulation Date: {sim_date.strftime('%Y-%m-%d')}\n"
        f"Agent Model: {agent_model}"
    ))

    async def run_evaluation():
        dataset_provider = CsvFinanceDatasetProvider(dataset_path) if dataset_path else None

        # Initialize components
        task_generator = DynamicTaskGenerator(dataset_provider=dataset_provider)
        evaluator = ComprehensiveEvaluator()
        if purple_base_url:
            agent = PurpleHTTPAgentClient(
                base_url=purple_base_url,
                agent_id="purple-agent",
                model=agent_model,
            )
        else:
            agent = MockAgentClient(agent_id="test-agent", model=agent_model)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Generate task
            progress.add_task(description="Generating dynamic task...", total=None)
            chosen_id = task_id
            if task_id.lower() == "random":
                candidates = task_generator.fab_dataset.questions
                if difficulty:
                    candidates = [q for q in candidates if q.difficulty in difficulty]
                if not candidates:
                    console.print("[red]No tasks match the given filters[/red]")
                    raise typer.Exit(1)
                chosen_id = random.choice(candidates).template_id

            task = await task_generator.generate_task(chosen_id, sim_date)

            if not task:
                console.print(f"[red]Error: Task template '{chosen_id}' not found[/red]")
                raise typer.Exit(1)

            # Run evaluation
            progress.add_task(description="Running evaluation...", total=None)
            result = await evaluator.run_full_evaluation(
                task=task,
                agent_client=agent,
                conduct_debate=not no_debate,
            )

            # Get agent response for report
            agent_response = await agent.process_task(task)

        return task, agent_response, result

    task, agent_response, result = asyncio.run(run_evaluation())

    # Generate output
    if output_format == "markdown":
        report = EvaluationReporter.generate_markdown_report(task, agent_response, result)
    elif output_format == "json":
        import json
        report = json.dumps(EvaluationReporter.generate_json_report(result), indent=2)
    else:
        report = EvaluationReporter.generate_summary(result)

    # Output
    if output_file:
        output_file.write_text(report)
        console.print(f"[green]Report saved to {output_file}[/green]")
    else:
        console.print(report)

    # Show Alpha Score prominently
    console.print(Panel.fit(
        f"[bold green]Alpha Score: {result.alpha_score.score:.2f}[/bold green]",
        title="Final Result"
    ))


@app.command()
def list_tasks():
    """
    List available FAB task templates.
    """
    dataset = FABDataset.load_sample_questions()

    table = Table(title="Available FAB Task Templates")
    table.add_column("ID", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Difficulty", style="yellow")
    table.add_column("Requires Code", style="red")

    for q in dataset.questions:
        table.add_row(
            q.template_id,
            q.category.value,
            q.difficulty.value,
            "Yes" if q.requires_code_execution else "No"
        )

    console.print(table)


@app.command()
def generate_task(
    task_id: str = typer.Argument(help="FAB question template ID"),
    simulation_date: str = typer.Option(
        None,
        "--date", "-d",
        help="Simulation date (YYYY-MM-DD)"
    ),
    dataset_path: Optional[Path] = typer.Option(
        None,
        "--dataset-path",
        help="Path to external CSV dataset (e.g., data/public.csv)"
    ),
    difficulty: Optional[list[TaskDifficulty]] = typer.Option(
        None,
        "--difficulty",
        help="Optional difficulty filter when selecting a random task",
    ),
):
    """
    Generate a dynamic task variant without running evaluation.
    """
    if simulation_date:
        sim_date = datetime.strptime(simulation_date, "%Y-%m-%d")
    else:
        sim_date = datetime(datetime.now().year - 1, 1, 1)

    async def generate():
        dataset_provider = CsvFinanceDatasetProvider(dataset_path) if dataset_path else None
        generator = DynamicTaskGenerator(dataset_provider=dataset_provider)

        chosen_id = task_id
        if task_id.lower() == "random":
            candidates = generator.fab_dataset.questions
            if difficulty:
                candidates = [q for q in candidates if q.difficulty in difficulty]
            if not candidates:
                console.print("[red]No tasks match the given filters[/red]")
                raise typer.Exit(1)
            chosen_id = random.choice(candidates).template_id

        return await generator.generate_task(chosen_id, sim_date)

    task = asyncio.run(generate())

    if not task:
        console.print(f"[red]Error: Task template '{task_id}' not found[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold cyan]Generated Task[/bold cyan]\n\n"
        f"[bold]ID:[/bold] {task.question_id}\n"
        f"[bold]Category:[/bold] {task.category.value}\n"
        f"[bold]Ticker:[/bold] {task.ticker}\n"
        f"[bold]Fiscal Year:[/bold] {task.fiscal_year}\n"
        f"[bold]Simulation Date:[/bold] {task.simulation_date.strftime('%Y-%m-%d')}\n\n"
        f"[bold]Question:[/bold]\n{task.question}\n\n"
        f"[bold]Requires Code:[/bold] {'Yes' if task.requires_code_execution else 'No'}"
    ))


@app.command()
def batch_evaluate(
    count: int = typer.Option(
        10,
        "--count", "-n",
        help="Number of tasks to evaluate"
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category", "-c",
        help="Filter by category"
    ),
    difficulty: Optional[str] = typer.Option(
        None,
        "--difficulty",
        help="Filter by difficulty"
    ),
    simulation_date: str = typer.Option(
        None,
        "--date", "-d",
        help="Simulation date (YYYY-MM-DD)"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--file", "-f",
        help="Output file for results (JSON)"
    ),
):
    """
    Run batch evaluation on multiple tasks.
    """
    if simulation_date:
        sim_date = datetime.strptime(simulation_date, "%Y-%m-%d")
    else:
        sim_date = datetime(datetime.now().year - 1, 1, 1)

    # Parse filters
    categories = None
    if category:
        try:
            categories = [TaskCategory(category)]
        except ValueError:
            console.print(f"[red]Invalid category: {category}[/red]")
            raise typer.Exit(1)

    difficulties = None
    if difficulty:
        try:
            difficulties = [TaskDifficulty(difficulty)]
        except ValueError:
            console.print(f"[red]Invalid difficulty: {difficulty}[/red]")
            raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold blue]Batch Evaluation[/bold blue]\n"
        f"Tasks: {count}\n"
        f"Category: {category or 'All'}\n"
        f"Difficulty: {difficulty or 'All'}\n"
        f"Simulation Date: {sim_date.strftime('%Y-%m-%d')}"
    ))

    async def run_batch():
        generator = DynamicTaskGenerator()
        evaluator = ComprehensiveEvaluator()
        agent = MockAgentClient(agent_id="batch-agent", model="gpt-4o")

        tasks = await generator.generate_task_batch(
            count=count,
            simulation_date=sim_date,
            categories=categories,
            difficulties=difficulties,
        )

        results = []
        with Progress(console=console) as progress:
            eval_task = progress.add_task("Evaluating...", total=len(tasks))

            for task in tasks:
                result = await evaluator.run_full_evaluation(
                    task=task,
                    agent_client=agent,
                    conduct_debate=True,
                )
                results.append(result)
                progress.advance(eval_task)

        return results

    results = asyncio.run(run_batch())

    # Summary table
    table = Table(title="Batch Evaluation Results")
    table.add_column("Task ID", style="cyan")
    table.add_column("Alpha Score", style="green")
    table.add_column("Role Score", style="yellow")
    table.add_column("Debate", style="magenta")
    table.add_column("Cost", style="red")

    total_alpha = 0
    for r in results:
        table.add_row(
            r.task_id[:30],
            f"{r.alpha_score.score:.2f}",
            f"{r.role_score.total:.1f}",
            f"{r.debate_result.debate_multiplier}x",
            f"${r.cost_breakdown.total_cost_usd:.4f}",
        )
        total_alpha += r.alpha_score.score

    console.print(table)

    avg_alpha = total_alpha / len(results) if results else 0
    console.print(f"\n[bold green]Average Alpha Score: {avg_alpha:.2f}[/bold green]")

    # Save to file if specified
    if output_file:
        import json
        output_data = {
            "batch_info": {
                "count": len(results),
                "simulation_date": sim_date.isoformat(),
                "average_alpha_score": avg_alpha,
            },
            "results": [EvaluationReporter.generate_json_report(r) for r in results],
        }
        output_file.write_text(json.dumps(output_data, indent=2))
        console.print(f"[green]Results saved to {output_file}[/green]")


@app.command()
def version():
    """
    Show version information.
    """
    console.print(Panel.fit(
        "[bold blue]CIO-Agent FAB++ Evaluator[/bold blue]\n"
        "Version: 1.0.0\n"
        "Team: AgentBusters\n"
        "Competition: AgentBeats Finance Track\n\n"
        "Dynamic Finance Agent Benchmark for evaluating\n"
        "financial analysis agents through adversarial testing."
    ))


if __name__ == "__main__":
    app()
