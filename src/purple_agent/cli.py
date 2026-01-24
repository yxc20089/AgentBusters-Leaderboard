"""
CLI for Purple Agent

Provides command-line interface for running the Purple Agent
A2A server and performing direct analysis tasks.
"""

import os
import asyncio
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="purple-agent",
    help="Purple Finance Agent - A2A-compliant finance analysis agent for AgentBeats",
)
console = Console()

# Load environment variables from .env if present so CLI picks up LLM and keys
load_dotenv()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8001, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    card_url: Optional[str] = typer.Option(
        None, "--card-url", help="Public URL for agent card (for container networking)"
    ),
    simulation_date: Optional[str] = typer.Option(
        None, "--simulation-date", "-d", help="Simulation date (YYYY-MM-DD)"
    ),
):
    """
    Start the Purple Agent A2A server.

    The server implements the A2A protocol and can be discovered
    by Green Agents for evaluation.
    """
    from purple_agent.server import create_app
    import uvicorn

    # Parse simulation date
    sim_date = None
    if simulation_date:
        try:
            sim_date = datetime.fromisoformat(simulation_date)
        except ValueError:
            console.print(f"[red]Invalid date format: {simulation_date}[/red]")
            raise typer.Exit(1)

    # Get API keys from environment
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("LLM_MODEL")
    temperature = os.environ.get("PURPLE_LLM_TEMPERATURE", "0.0")

    # Create and run app
    console.print(Panel.fit(
        f"[bold blue]Purple Finance Agent[/bold blue]\n"
        f"Host: {host}:{port}\n"
        f"LLM: {'OpenAI' if openai_key else 'Anthropic' if anthropic_key else 'None (fallback mode)'}\n"
        f"Temperature: {temperature} (set PURPLE_LLM_TEMPERATURE to change)\n"
        f"Simulation Date: {sim_date or 'None (live data)'}"
    ))

    app_instance = create_app(
        host=host,
        port=port,
        card_url=card_url,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        model=model,
        simulation_date=sim_date,
    )

    console.print(f"\n[green]Starting A2A server at http://{host}:{port}[/green]")
    console.print(f"[dim]Agent Card: http://{host}:{port}/.well-known/agent.json[/dim]\n")

    uvicorn.run(
        app_instance,
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def analyze(
    question: str = typer.Argument(..., help="The analysis question"),
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Stock ticker"),
    simulation_date: Optional[str] = typer.Option(
        None, "--simulation-date", "-d", help="Simulation date (YYYY-MM-DD)"
    ),
):
    """
    Perform a direct financial analysis.

    This runs analysis locally without starting the A2A server.
    """
    from purple_agent.agent import create_agent

    # Parse simulation date
    sim_date = None
    if simulation_date:
        try:
            sim_date = datetime.fromisoformat(simulation_date)
        except ValueError:
            console.print(f"[red]Invalid date format: {simulation_date}[/red]")
            raise typer.Exit(1)

    # Get API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("LLM_MODEL")

    async def run_analysis():
        agent = await create_agent(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            model=model,
            simulation_date=sim_date,
        )

        console.print(Panel.fit(
            f"[bold cyan]Question:[/bold cyan] {question}\n"
            f"[dim]Ticker: {ticker or 'Auto-detect'}[/dim]"
        ))

        console.print("\n[yellow]Analyzing...[/yellow]\n")

        analysis = await agent.analyze(question, ticker)

        console.print(Panel(analysis, title="Analysis Result", border_style="green"))

    asyncio.run(run_analysis())


@app.command()
def info(
    ticker: str = typer.Argument(..., help="Stock ticker symbol"),
):
    """
    Get comprehensive stock information via MCP servers.
    """
    from purple_agent.mcp_toolkit import MCPToolkit

    async def get_info():
        toolkit = MCPToolkit()
        data = await toolkit.get_comprehensive_analysis(ticker)

        # Display quote info
        if "quote" in data and "error" not in data["quote"]:
            quote = data["quote"]
            table = Table(title=f"{ticker} - Stock Quote")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Price", f"${quote.get('current_price', 'N/A')}")
            table.add_row("Market Cap", f"${quote.get('market_cap', 'N/A'):,}" if quote.get('market_cap') else "N/A")
            table.add_row("P/E Ratio", str(quote.get("pe_ratio", "N/A")))
            table.add_row("52W High", f"${quote.get('fifty_two_week_high', 'N/A')}")
            table.add_row("52W Low", f"${quote.get('fifty_two_week_low', 'N/A')}")
            table.add_row("Analyst Rating", str(quote.get("analyst_rating", "N/A")))

            console.print(table)

        # Display company info from SEC EDGAR
        if "company_info" in data and "error" not in data["company_info"]:
            company = data["company_info"]
            table = Table(title="Company Information (SEC EDGAR)")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Name", str(company.get("name", "N/A")))
            table.add_row("CIK", str(company.get("cik", "N/A")))
            table.add_row("SIC", str(company.get("sic", "N/A")))

            console.print(table)

        # Display key statistics
        if "statistics" in data and "error" not in data["statistics"]:
            stats = data["statistics"]
            table = Table(title="Key Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            if stats.get("beta"):
                table.add_row("Beta", f"{stats['beta']:.2f}")
            if stats.get("profit_margin"):
                table.add_row("Profit Margin", f"{stats['profit_margin']*100:.1f}%")
            if stats.get("return_on_equity"):
                table.add_row("ROE", f"{stats['return_on_equity']*100:.1f}%")

            console.print(table)

        # Display recent filing
        if "recent_filing" in data and "error" not in data["recent_filing"]:
            filing = data["recent_filing"]
            table = Table(title="Recent SEC Filing")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Form", str(filing.get("form_type", "N/A")))
            table.add_row("Date", str(filing.get("filing_date", "N/A")))
            table.add_row("Accession", str(filing.get("accession_number", "N/A")))

            console.print(table)

    asyncio.run(get_info())


@app.command()
def card():
    """
    Display the Agent Card.
    """
    from purple_agent.card import get_agent_card
    import json

    agent_card = get_agent_card()
    card_dict = agent_card.model_dump(exclude_none=True)

    console.print(Panel(
        json.dumps(card_dict, indent=2),
        title="Agent Card",
        border_style="blue"
    ))


if __name__ == "__main__":
    app()
