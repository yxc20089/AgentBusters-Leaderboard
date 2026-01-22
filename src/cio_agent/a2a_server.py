"""
A2A Server for Green Agent (CIO-Agent FAB++ Evaluator)

This is the main entry point for the Green Agent A2A server.
It exposes the FAB++ evaluation capabilities via the A2A protocol,
allowing the AgentBeats platform to run assessments.

Usage:
    uv run src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9009
    
    # With synthetic questions:
    uv run src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9009 \\
        --synthetic-questions data/synthetic_questions/questions.json
    
    # Or with Docker:
    docker run -p 9009:9009 ghcr.io/your-org/cio-agent-green:latest --host 0.0.0.0

    # Enable LLM grading for dataset evaluators:
    uv run src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9009 --eval-llm \
        --eval-llm-model gpt-4o-mini --eval-llm-temperature 0.0

The server accepts:
    --host: Host address to bind to (default: 127.0.0.1)
    --port: Port to listen on (default: 9009)
    --card-url: URL to advertise in the agent card (optional)
    --synthetic-questions: Path to synthetic questions JSON file (optional)
"""

import argparse
import json
import logging
import os
from pathlib import Path
import uvicorn
from sqlalchemy.ext.asyncio import create_async_engine

from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import DatabaseTaskStore, InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from cio_agent.green_executor import GreenAgentExecutor

# Load environment variables from .env file
load_dotenv()


def load_synthetic_questions(file_path: str) -> list[dict]:
    """Load synthetic questions from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Synthetic questions file not found: {file_path}")
    
    data = json.loads(path.read_text())
    questions = data.get("questions", [])
    if not questions:
        raise ValueError(f"No questions found in {file_path}")
    
    print(f"Loaded {len(questions)} synthetic questions from {file_path}")
    return questions


def main():
    """Main entry point for the Green Agent A2A server."""
    parser = argparse.ArgumentParser(description="Run the CIO-Agent Green Agent A2A server.")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9009,
        help="Port to bind the server"
    )
    parser.add_argument(
        "--card-url",
        type=str,
        help="URL to advertise in the agent card"
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        help="Path to evaluation config YAML file (recommended for multi-dataset evaluation)"
    )
    parser.add_argument(
        "--synthetic-questions",
        type=str,
        help="Path to synthetic questions JSON file (legacy, use --eval-config instead)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="synthetic",
        choices=["synthetic", "bizfinbench", "public_csv"],
        help="Type of dataset (legacy, use --eval-config instead)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to dataset directory or file (legacy, use --eval-config instead)"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        help="For BizFinBench, the specific task type (legacy, use --eval-config instead)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "cn"],
        help="Language for BizFinBench dataset (legacy, use --eval-config instead)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples (legacy, use --eval-config instead)"
    )
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--eval-llm",
        action="store_true",
        help="Enable LLM grading for dataset evaluators (bizfinbench/public_csv)"
    )
    eval_group.add_argument(
        "--no-eval-llm",
        action="store_true",
        help="Disable LLM grading for dataset evaluators (bizfinbench/public_csv)"
    )
    parser.add_argument(
        "--eval-llm-model",
        type=str,
        help="Model override for LLM grading (e.g., gpt-4o-mini)"
    )
    parser.add_argument(
        "--eval-llm-temperature",
        type=float,
        help="Temperature for LLM grading (default 0.0)"
    )
    parser.add_argument(
        "--store-predicted",
        action="store_true",
        help="Store predicted outputs in evaluation results"
    )
    trunc_group = parser.add_mutually_exclusive_group()
    trunc_group.add_argument(
        "--truncate-predicted",
        action="store_true",
        help="Truncate predicted outputs in evaluation results"
    )
    trunc_group.add_argument(
        "--no-truncate-predicted",
        action="store_true",
        help="Do not truncate predicted outputs in evaluation results"
    )
    parser.add_argument(
        "--predicted-max-chars",
        type=int,
        help="Maximum characters for predicted outputs when truncation is enabled"
    )
    args = parser.parse_args()

    # Validate configuration
    eval_config = None
    if args.eval_config:
        # Config file mode (recommended)
        from pathlib import Path
        if not Path(args.eval_config).exists():
            print(f"Error: Config file not found: {args.eval_config}")
            return 1
        eval_config = args.eval_config
        print(f"Using evaluation config: {args.eval_config}")
    elif args.dataset_type in ("bizfinbench", "public_csv") and not args.dataset_path:
        print(f"Error: --dataset-path is required for {args.dataset_type}")
        return 1

    eval_use_llm = None
    if args.eval_llm:
        eval_use_llm = True
    elif args.no_eval_llm:
        eval_use_llm = False

    truncate_predicted = None
    if args.truncate_predicted:
        truncate_predicted = True
    elif args.no_truncate_predicted:
        truncate_predicted = False

    # Load synthetic questions if provided (legacy mode)
    synthetic_questions = None
    if args.synthetic_questions:
        try:
            synthetic_questions = load_synthetic_questions(args.synthetic_questions)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading synthetic questions: {e}")
            return 1

    # Define agent skills
    skill = AgentSkill(
        id="fab-plus-plus-evaluation",
        name="FAB++ Finance Agent Benchmark",
        description=(
            "Evaluates finance agents using the FAB++ (Finance Agent Benchmark Plus Plus) "
            "framework. Assesses agents on macro analysis, fundamental accuracy, execution "
            "quality, and adversarial robustness. Provides comprehensive Alpha Scores."
        ),
        tags=[
            "finance",
            "evaluation",
            "benchmark",
            "agent-assessment",
            "adversarial-debate",
        ],
        examples=[
            "Evaluate a finance agent's ability to analyze NVIDIA Q3 earnings",
            "Test agent robustness with adversarial counter-arguments",
            "Assess fundamental data accuracy and macro thesis quality",
        ]
    )
    crypto_skill = AgentSkill(
        id="agentbusters-crypto-benchmark",
        name="AgentBusters Crypto Trading Benchmark",
        description=(
            "Evaluates crypto trading agents on multi-step market scenarios. "
            "Scores robustness across baseline, noisy, adversarial, and "
            "meta-consistency evaluations with portfolio-level metrics."
        ),
        tags=[
            "crypto",
            "trading",
            "evaluation",
            "benchmark",
            "multi-round",
        ],
        examples=[
            "Evaluate a BTCUSDT strategy on a volatile drawdown window",
            "Test robustness under noisy price perturbations",
            "Assess consistency across transformed market regimes",
        ],
    )

    # Determine the advertised URL
    # Note: 0.0.0.0 is a bind address, not connectable. Use 127.0.0.1 for local testing.
    advertised_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    card_url = args.card_url or f"http://{advertised_host}:{args.port}/"

    # Create agent card
    agent_card = AgentCard(
        name="CIO-Agent FAB++ Evaluator",
        description=(
            "A Green Agent for the AgentBeats platform that evaluates finance agents "
            "using the FAB++ (Finance Agent Benchmark Plus Plus) methodology. "
            "Tests agents on earnings analysis, SEC filing interpretation, numerical "
            "reasoning, investment recommendations, and crypto trading robustness testing."
        ),
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill, crypto_skill]
    )

    # Create request handler with executor
    logger = logging.getLogger(__name__)
    database_url = os.getenv("DATABASE_URL")

    task_store = None
    if database_url:
        logger.info(f"Using database task store: {database_url}")
        try:
            engine = create_async_engine(database_url)
            task_store = DatabaseTaskStore(engine)
        except Exception as e:
            logger.warning(f"Failed to initialize database task store: {e}")
            logger.warning("Falling back to in-memory task store")

    if task_store is None:
        logger.info("Using in-memory task store")
        task_store = InMemoryTaskStore()
    
    request_handler = DefaultRequestHandler(
        agent_executor=GreenAgentExecutor(
            eval_config=eval_config,
            synthetic_questions=synthetic_questions,
            dataset_type=args.dataset_type,
            dataset_path=args.dataset_path,
            task_type=args.task_type,
            language=args.language,
            limit=args.limit,
            eval_use_llm=eval_use_llm,
            eval_llm_model=args.eval_llm_model,
            eval_llm_temperature=args.eval_llm_temperature,
            store_predicted=args.store_predicted,
            truncate_predicted=truncate_predicted,
            predicted_max_chars=args.predicted_max_chars,
        ),
        task_store=task_store,
    )
    
    # Create A2A application
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print(f"Starting CIO-Agent Green Agent A2A server...")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Agent Card URL: {agent_card.url}")
    print(f"  Agent Card: http://{args.host}:{args.port}/.well-known/agent.json")
    
    if eval_config:
        # Config-based mode (recommended)
        from cio_agent.eval_config import EvaluationConfig
        config = EvaluationConfig.from_yaml(eval_config)
        print(f"  Mode: Config-based ({config.name})")
        print(f"  Datasets: {len(config.datasets)}")
        for ds in config.datasets:
            print(f"    - {ds.type}")
        print(f"  Sampling: {config.sampling.strategy}")
        if config.sampling.total_limit:
            print(f"  Total Limit: {config.sampling.total_limit}")
        if eval_use_llm is not None:
            print(f"  LLM Grading (CLI): {'enabled' if eval_use_llm else 'disabled'}")
        elif config.llm_eval and (
            config.llm_eval.enabled is not None
            or config.llm_eval.model
            or config.llm_eval.temperature is not None
        ):
            print(f"  LLM Grading (config): {config.llm_eval.enabled}")
    else:
        # Legacy mode
        print(f"  Mode: Legacy")
        print(f"  Dataset Type: {args.dataset_type}")
        if args.dataset_path:
            print(f"  Dataset Path: {args.dataset_path}")
        if args.task_type:
            print(f"  Task Type: {args.task_type}")
        if args.limit:
            print(f"  Limit: {args.limit} examples")
        if synthetic_questions:
            print(f"  Synthetic Questions: {len(synthetic_questions)} loaded")
        if eval_use_llm is not None:
            print(f"  LLM Grading (CLI): {'enabled' if eval_use_llm else 'disabled'}")
    
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
