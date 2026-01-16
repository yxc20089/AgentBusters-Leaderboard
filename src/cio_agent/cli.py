"""
CLI for CIO-Agent FAB++ Evaluator.

Provides command-line interface for:
- Running evaluations
- Generating tasks
- Viewing reports
"""

import os
import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from cio_agent.task_generator import DynamicTaskGenerator, FABDataset
from cio_agent.evaluator import ComprehensiveEvaluator, EvaluationReporter
from cio_agent.models import TaskCategory, TaskDifficulty, Task, GroundTruth, TaskRubric
from cio_agent.data_providers.csv_provider import CsvFinanceDatasetProvider
from cio_agent.a2a_client import PurpleHTTPAgentClient

# Load environment variables from .env so default model picks up LLM_MODEL/OPENAI_API_BASE
load_dotenv()

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
        os.environ.get("LLM_MODEL", "gpt-4o"),
        "--model", "-m",
        help="Agent model to simulate (defaults to LLM_MODEL env var if set)"
    ),
    purple_base_url: str = typer.Option(
        ...,
        "--purple-endpoint",
        help="Purple agent base URL (e.g., http://purple-agent:8010 or http://localhost:8010)",
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
        agent = PurpleHTTPAgentClient(
            base_url=purple_base_url,
            agent_id="purple-agent",
            model=agent_model,
        )

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
    purple_base_url: str = typer.Option(
        ...,
        "--purple-endpoint",
        help="Purple agent base URL (e.g., http://purple-agent:8010 or http://localhost:8010)",
        envvar="PURPLE_ENDPOINT",
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
        agent = PurpleHTTPAgentClient(
            base_url=purple_base_url,
            agent_id="purple-agent",
            model="purple-http",
        )

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


# =============================================================================
# SYNTHETIC BENCHMARK COMMANDS
# =============================================================================

@app.command()
def harvest(
    tickers: Optional[str] = typer.Option(
        None,
        "--tickers", "-t",
        help="Comma-separated list of tickers (defaults to full universe)"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Re-fetch data even if it exists"
    ),
):
    """
    Harvest financial data from AlphaVantage into the Financial Lake.
    
    Populates local cache with fundamental data for synthetic question generation.
    Requires ALPHAVANTAGE_API_KEY in environment.
    """
    from cio_agent.financial_lake import FinancialLake, ALL_TICKERS
    
    ticker_list = None
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
    else:
        ticker_list = ALL_TICKERS
    
    console.print(Panel.fit(
        f"[bold blue]Financial Lake Harvester[/bold blue]\n"
        f"Tickers: {len(ticker_list)}\n"
        f"Force refresh: {force}"
    ))
    
    async def run_harvest():
        lake = FinancialLake()
        return await lake.harvest(ticker_list, force=force)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description="Harvesting financial data...", total=None)
        results = asyncio.run(run_harvest())
    
    # Summary
    success = sum(results.values())
    failed = len(results) - success
    
    table = Table(title="Harvest Results")
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="green")
    
    table.add_row("Success", str(success))
    table.add_row("Failed", str(failed))
    
    console.print(table)
    
    if failed > 0:
        failed_tickers = [t for t, ok in results.items() if not ok]
        console.print(f"[yellow]Failed tickers: {', '.join(failed_tickers[:10])}...[/yellow]")


@app.command()
def generate_synthetic(
    count: int = typer.Option(
        10,
        "--count", "-n",
        help="Number of questions to generate"
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category", "-c",
        help="Filter by category (e.g., 'Quantitative Retrieval')"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for generated questions (JSON)"
    ),
    respect_weights: bool = typer.Option(
        True,
        "--respect-weights/--even-distribution",
        help="Use FAB benchmark category weights"
    ),
):
    """
    Generate synthetic benchmark questions using the Financial Lake.
    
    Uses the Generator-Verifier-Refiner architecture to create
    high-quality questions with pre-calculated ground truth.
    """
    from cio_agent.synthetic_generator import SyntheticTaskGenerator
    from cio_agent.financial_lake import FinancialLake
    
    console.print(Panel.fit(
        f"[bold blue]Synthetic Question Generator[/bold blue]\n"
        f"Count: {count}\n"
        f"Category: {category or 'All (weighted)'}\n"
        f"Respect weights: {respect_weights}"
    ))
    
    # Parse category filter
    categories = None
    if category:
        try:
            categories = [TaskCategory(category)]
        except ValueError:
            console.print(f"[red]Invalid category: {category}[/red]")
            console.print("Valid categories: " + ", ".join([c.value for c in TaskCategory]))
            raise typer.Exit(1)
    
    lake = FinancialLake()
    available = lake.get_available_tickers()
    
    if not available:
        console.print("[red]No data in Financial Lake. Run 'cio-agent harvest' first.[/red]")
        raise typer.Exit(1)
    
    console.print(f"[dim]Using {len(available)} tickers from Financial Lake[/dim]")
    
    generator = SyntheticTaskGenerator(financial_lake=lake)
    questions = generator.generate_batch(
        count=count,
        categories=categories,
        respect_weights=respect_weights and not category,
    )
    
    # Display summary
    table = Table(title=f"Generated {len(questions)} Questions")
    table.add_column("ID", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Difficulty", style="yellow")
    table.add_column("Ticker", style="magenta")
    table.add_column("Question Preview", style="white")
    
    for q in questions[:20]:  # Show first 20
        table.add_row(
            q.question_id,
            q.category.value[:20],
            q.difficulty.value,
            q.ticker,
            q.question[:40] + "..."
        )
    
    if len(questions) > 20:
        table.add_row("...", f"({len(questions) - 20} more)", "", "", "")
    
    console.print(table)
    
    # Category distribution
    cat_dist = {}
    for q in questions:
        cat_dist[q.category.value] = cat_dist.get(q.category.value, 0) + 1
    
    dist_table = Table(title="Category Distribution")
    dist_table.add_column("Category", style="cyan")
    dist_table.add_column("Count", style="green")
    dist_table.add_column("%", style="yellow")
    
    for cat, cnt in sorted(cat_dist.items(), key=lambda x: -x[1]):
        dist_table.add_row(cat, str(cnt), f"{cnt/len(questions)*100:.1f}%")
    
    console.print(dist_table)
    
    # Save to file
    if output_file:
        import json
        output_data = {
            "generated_at": datetime.now().isoformat(),
            "count": len(questions),
            "questions": [q.model_dump() for q in questions],
        }
        try:
            output_file.write_text(json.dumps(output_data, indent=2, default=str))
            console.print(f"[green]Questions saved to {output_file}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to save questions to {output_file}: {e}[/red]")


@app.command()
def verify_questions(
    input_file: Path = typer.Argument(help="JSON file with generated questions"),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for verification report (JSON)"
    ),
    tolerance: float = typer.Option(
        0.05,
        "--tolerance",
        help="Numerical tolerance for answer comparison (0.05 = 5%)"
    ),
):
    """
    Verify synthetic questions for solvability.
    
    Runs the Verifier to check that questions are answerable
    and flags any that need refinement.
    """
    import json
    from cio_agent.synthetic_generator import SyntheticQuestion
    from cio_agent.verifier import QuestionVerifier, VerificationResult
    
    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    # Load questions
    data = json.loads(input_file.read_text())
    questions = [SyntheticQuestion(**q) for q in data.get("questions", [])]
    
    console.print(Panel.fit(
        f"[bold blue]Question Verifier[/bold blue]\n"
        f"Questions: {len(questions)}\n"
        f"Tolerance: {tolerance*100:.1f}%"
    ))
    
    verifier = QuestionVerifier(numerical_tolerance=tolerance)
    results = verifier.verify_batch(questions)
    
    # Summary
    table = Table(title="Verification Results")
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("%", style="yellow")
    
    table.add_row(
        "[green]Accept[/green]",
        str(results["accept"]),
        f"{results['accept']/results['total']*100:.1f}%"
    )
    table.add_row(
        "[yellow]Refine[/yellow]",
        str(results["refine"]),
        f"{results['refine']/results['total']*100:.1f}%"
    )
    table.add_row(
        "[red]Reject[/red]",
        str(results["reject"]),
        f"{results['reject']/results['total']*100:.1f}%"
    )
    
    console.print(table)
    
    # Show issues for questions needing attention
    refine_reports = [r for r in results["reports"] if r.result != VerificationResult.ACCEPT]
    if refine_reports[:5]:
        console.print("\n[bold]Questions needing attention:[/bold]")
        for r in refine_reports[:5]:
            console.print(f"  [yellow]{r.question_id}[/yellow]: {', '.join(r.issues)}")
    
    # Save report
    if output_file:
        report_data = {
            "verified_at": datetime.now().isoformat(),
            "summary": {
                "total": results["total"],
                "accept": results["accept"],
                "reject": results["reject"],
                "refine": results["refine"],
                "accept_rate": results["accept_rate"],
            },
            "reports": [r.model_dump() for r in results["reports"]],
        }
        try:
            output_file.write_text(json.dumps(report_data, indent=2, default=str))
            console.print(f"[green]Report saved to {output_file}[/green]")
        except OSError as e:
            console.print(f"[red]Failed to save report to {output_file}: {e}[/red]")


@app.command()
def lake_status():
    """
    Show status of the Financial Lake data.
    """
    from cio_agent.financial_lake import FinancialLake, TICKER_UNIVERSE
    
    lake = FinancialLake()
    available = lake.get_available_tickers()
    
    console.print(Panel.fit(
        f"[bold blue]Financial Lake Status[/bold blue]\n"
        f"Total tickers: {len(available)}"
    ))
    
    # By sector
    table = Table(title="Data by Sector")
    table.add_column("Sector", style="cyan")
    table.add_column("Available", style="green")
    table.add_column("Total", style="yellow")
    table.add_column("Coverage", style="magenta")
    
    for sector, tickers in TICKER_UNIVERSE.items():
        available_in_sector = lake.get_tickers_by_sector(sector)
        coverage = len(available_in_sector) / len(tickers) * 100
        table.add_row(
            sector.replace("_", " ").title(),
            str(len(available_in_sector)),
            str(len(tickers)),
            f"{coverage:.0f}%"
        )
    
    console.print(table)
    
    if not available:
        console.print("\n[yellow]Financial Lake is empty. Run 'cio-agent harvest' to populate.[/yellow]")


@app.command()
def evaluate_synthetic(
    questions_file: Path = typer.Argument(
        ...,
        help="Path to synthetic questions JSON file"
    ),
    purple_base_url: str = typer.Option(
        ...,
        "--purple-endpoint",
        help="Purple agent base URL (e.g., http://localhost:9110)",
        envvar="PURPLE_ENDPOINT",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit", "-n",
        help="Maximum number of questions to evaluate"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for results (JSON)"
    ),
    no_debate: bool = typer.Option(
        False,
        "--no-debate",
        help="Skip adversarial debate phase"
    ),
):
    """
    Evaluate synthetic questions against the Purple Agent.
    
    Loads questions from a JSON file generated by 'generate-synthetic'
    and runs the full evaluation pipeline including debate.
    
    Example:
        cio-agent evaluate-synthetic data/synthetic_questions/questions.json \\
            --purple-endpoint http://localhost:9110 \\
            --output results.json
    """
    import json
    from cio_agent.a2a_client import PurpleHTTPAgentClient
    from cio_agent.evaluator import ComprehensiveEvaluator, EvaluationReporter
    from cio_agent.synthetic_generator import SyntheticQuestion
    
    if not questions_file.exists():
        console.print(f"[red]File not found: {questions_file}[/red]")
        raise typer.Exit(1)
    
    # Load questions
    try:
        data = json.loads(questions_file.read_text())
        raw_questions = data.get("questions", [])
    except (json.JSONDecodeError, KeyError) as e:
        console.print(f"[red]Invalid questions file: {e}[/red]")
        raise typer.Exit(1)
    
    if not raw_questions:
        console.print("[red]No questions found in file[/red]")
        raise typer.Exit(1)
    
    # Apply limit
    if limit:
        raw_questions = raw_questions[:limit]
    
    console.print(Panel.fit(
        f"[bold blue]Synthetic Question Evaluation[/bold blue]\n"
        f"Questions file: {questions_file}\n"
        f"Question count: {len(raw_questions)}\n"
        f"Purple endpoint: {purple_base_url}\n"
        f"Debate: {'Disabled' if no_debate else 'Enabled'}"
    ))
    
    # Convert SyntheticQuestion to Task format
    def convert_to_task(sq_data: dict) -> Task:
        """Convert synthetic question dict to Task object."""
        # Handle category enum
        category_value = sq_data.get("category", "Quantitative Retrieval")
        try:
            category = TaskCategory(category_value)
        except ValueError:
            category = TaskCategory.QUANTITATIVE_RETRIEVAL
        
        # Handle difficulty enum
        difficulty_value = sq_data.get("difficulty", "medium")
        try:
            difficulty = TaskDifficulty(difficulty_value)
        except ValueError:
            difficulty = TaskDifficulty.MEDIUM
        
        # Build ground truth with required fields
        from cio_agent.models import FinancialData
        ground_truth = GroundTruth(
            macro_thesis=str(sq_data.get("ground_truth_formatted", "Evaluate the analysis")),
            key_themes=sq_data.get("calculation_steps", []),
            expected_recommendation=str(sq_data.get("ground_truth_formatted", "")),
            financials=FinancialData(),
        )
        
        # Build rubric from components
        rubric_data = sq_data.get("rubric", {})
        rubric_components = rubric_data.get("components", [])
        rubric = TaskRubric(
            criteria=[c.get("description", "") for c in rubric_components],
            max_score=rubric_data.get("max_score", 100),
        )
        
        from datetime import datetime
        return Task(
            question_id=sq_data.get("question_id", "SYN_UNKNOWN"),
            category=category,
            difficulty=difficulty,
            question=sq_data.get("question", ""),
            ticker=sq_data.get("ticker", "AAPL"),
            fiscal_year=sq_data.get("fiscal_year", 2024),
            simulation_date=datetime.now(),
            ground_truth=ground_truth,
            rubric=rubric,
            requires_code_execution=sq_data.get("requires_code_execution", False),
        )
    
    async def run_evaluation():
        evaluator = ComprehensiveEvaluator()
        agent = PurpleHTTPAgentClient(
            base_url=purple_base_url,
            agent_id="purple-agent",
            model="purple-http",
        )
        
        results = []
        with Progress(console=console) as progress:
            eval_task = progress.add_task("Evaluating...", total=len(raw_questions))
            
            for sq_data in raw_questions:
                task = convert_to_task(sq_data)
                
                try:
                    result = await evaluator.run_full_evaluation(
                        task=task,
                        agent_client=agent,
                        conduct_debate=not no_debate,
                    )
                    results.append(result)
                except Exception as e:
                    logger.error("evaluation_error", task_id=task.question_id, error=str(e))
                    console.print(f"[yellow]Warning: Failed to evaluate {task.question_id}: {e}[/yellow]")
                
                progress.advance(eval_task)
        
        return results
    
    results = asyncio.run(run_evaluation())
    
    if not results:
        console.print("[red]No evaluations completed successfully[/red]")
        raise typer.Exit(1)
    
    # Summary table
    table = Table(title="Synthetic Evaluation Results")
    table.add_column("Question ID", style="cyan")
    table.add_column("Category", style="blue")
    table.add_column("Alpha Score", style="green")
    table.add_column("Role Score", style="yellow")
    table.add_column("Debate", style="magenta")
    
    total_alpha = 0
    for r in results:
        table.add_row(
            r.task_id[:20],
            r.task_id.split("_")[1] if "_" in r.task_id else "?",
            f"{r.alpha_score.score:.2f}",
            f"{r.role_score.total:.1f}",
            f"{r.debate_result.debate_multiplier}x",
        )
        total_alpha += r.alpha_score.score
    
    console.print(table)
    
    avg_alpha = total_alpha / len(results) if results else 0
    console.print(f"\n[bold green]Average Alpha Score: {avg_alpha:.2f}[/bold green]")
    console.print(f"[bold]Evaluated: {len(results)}/{len(raw_questions)} questions[/bold]")
    
    # Save to file if specified
    if output_file:
        output_data = {
            "evaluation_info": {
                "questions_file": str(questions_file),
                "count": len(results),
                "average_alpha_score": avg_alpha,
            },
            "results": [EvaluationReporter.generate_json_report(r) for r in results],
        }
        output_file.write_text(json.dumps(output_data, indent=2))
        console.print(f"[green]Results saved to {output_file}[/green]")


if __name__ == "__main__":
    app()

