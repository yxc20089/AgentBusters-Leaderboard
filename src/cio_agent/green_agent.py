"""
Green Agent Implementation for AgentBeats Platform

This is the core agent logic that orchestrates evaluation of Purple Agents.
It receives an assessment request with participant agent URLs and config,
then runs the FAB++ evaluation pipeline.

Supported modes:
    - config: Use YAML config file for multi-dataset evaluation
    - synthetic: Generated questions from JSON file
    - bizfinbench: HiThink BizFinBench.v2 dataset (single task type)
    - public_csv: FAB++ public.csv dataset
    - crypto: Crypto trading benchmark scenarios (config mode)
"""

import json
import os
from pathlib import Path
from typing import Any, Optional, List, Union
from pydantic import BaseModel
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from cio_agent.messenger import Messenger
from cio_agent.evaluator import ComprehensiveEvaluator, EvaluationReporter
from cio_agent.task_generator import DynamicTaskGenerator
from cio_agent.models import Task as FABTask, TaskCategory, TaskDifficulty, GroundTruth, FinancialData, TaskRubric
from cio_agent.eval_config import (
    EvaluationConfig,
    ConfigurableDatasetLoader,
    LoadedExample,
    create_default_config,
)
from cio_agent.agentbeats_results import format_and_save_results
from cio_agent.unified_scoring import UnifiedScorer, ScoreSection, DATASET_SECTION_MAP
from cio_agent.crypto_benchmark import CryptoTradingEvaluator, stable_seed

# Dataset providers (for legacy single-dataset mode)
from cio_agent.data_providers import BizFinBenchProvider, CsvFinanceDatasetProvider

# Dataset-specific evaluators
from evaluators import BizFinBenchEvaluator, PublicCsvEvaluator, OptionsEvaluator
from evaluators.gdpval_evaluator import GDPValEvaluator
from evaluators.llm_utils import build_llm_client, should_use_llm


class GreenAgent:
    """
    CIO-Agent Green Agent for FAB++ Finance Agent Benchmark.
    
    This agent evaluates Purple Agents on their financial analysis capabilities
    using the FAB++ evaluation framework.
    
    Initialization modes:
        1. Config-based (recommended): Pass eval_config for multi-dataset support
        2. Legacy single-dataset: Use dataset_type, dataset_path, etc.
        3. Synthetic: Use synthetic_questions list
    
    Required roles:
        - purple_agent: The finance agent being evaluated
        
    Config options:
        - num_tasks: Number of evaluation tasks (default: 1)
        - conduct_debate: Whether to run adversarial debate (default: True)
    """
    
    # Required participant roles
    required_roles: list[str] = ["purple_agent"]
    
    # Required config keys (optional ones will have defaults)
    required_config_keys: list[str] = []

    def validate_request(self, request: "EvalRequest") -> tuple[bool, str]:
        """Validate required roles and config keys for an evaluation request."""
        missing_roles = [role for role in self.required_roles if role not in request.participants]
        if missing_roles:
            return False, f"Missing roles: {', '.join(missing_roles)}"

        missing_keys = [key for key in self.required_config_keys if key not in request.config]
        if missing_keys:
            return False, f"Missing config keys: {', '.join(missing_keys)}"

        return True, "ok"


    def __init__(
        self,
        eval_config: Optional[Union[EvaluationConfig, str, Path]] = None,
        synthetic_questions: Optional[List[dict]] = None,
        dataset_type: str = "synthetic",
        dataset_path: Optional[str] = None,
        task_type: Optional[str] = None,
        language: str = "en",
        limit: Optional[int] = None,
        eval_use_llm: Optional[bool] = None,
        eval_llm_model: Optional[str] = None,
        eval_llm_temperature: Optional[float] = None,
        store_predicted: bool = False,
        truncate_predicted: Optional[bool] = None,
        predicted_max_chars: Optional[int] = None,
    ):
        """
        Initialize the Green Agent.
        
        Args:
            eval_config: Configuration for multi-dataset evaluation.
                        Can be EvaluationConfig, path to YAML file, or None.
            synthetic_questions: Optional list of synthetic questions to use
                                for evaluation. If provided, these will be used
                                instead of generating new tasks.
            dataset_type: Type of dataset to use ('synthetic', 'bizfinbench', 'public_csv')
            dataset_path: Path to dataset directory or file
            task_type: For BizFinBench, the specific task type to evaluate
            language: Language for BizFinBench ('en' or 'cn')
            limit: Optional limit on number of examples
            eval_use_llm: Optional override to enable/disable LLM grading
            eval_llm_model: Optional LLM model override for grading
            eval_llm_temperature: Optional temperature override for grading
            store_predicted: Whether to store predicted outputs in results
            truncate_predicted: Optional override to truncate predicted outputs
            predicted_max_chars: Optional max length for predicted outputs
        """
        self.messenger = Messenger()
        self.evaluator = ComprehensiveEvaluator()
        self.task_generator = DynamicTaskGenerator()
        self.synthetic_questions = synthetic_questions or []
        
        # Legacy dataset configuration
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.language = language
        self.limit = limit
        
        # Config-based multi-dataset support
        self.eval_config: Optional[EvaluationConfig] = None
        self.dataset_loader: Optional[ConfigurableDatasetLoader] = None
        self._loaded_examples: Optional[List[LoadedExample]] = None
        
        # Initialize based on provided mode
        self.dataset_provider = None
        self.dataset_evaluator = None
        self._examples = None  # Legacy cached examples
        
        # Priority: eval_config > single dataset > synthetic
        if eval_config is not None:
            # Config-based multi-dataset mode
            if isinstance(eval_config, (str, Path)):
                self.eval_config = EvaluationConfig.from_yaml(eval_config)
            else:
                self.eval_config = eval_config
            
            self.dataset_loader = ConfigurableDatasetLoader(self.eval_config)
            self._loaded_examples = self.dataset_loader.load()

        config_llm = self.eval_config.llm_eval if self.eval_config else None
        config_use_llm = config_llm.enabled if config_llm and config_llm.enabled is not None else None
        config_llm_model = config_llm.model if config_llm and config_llm.model else None
        config_llm_temp = (
            config_llm.temperature if config_llm and config_llm.temperature is not None else None
        )

        if eval_use_llm is not None:
            self.use_llm = eval_use_llm
        elif config_use_llm is not None:
            self.use_llm = config_use_llm
        else:
            self.use_llm = should_use_llm()

        self.llm_model = eval_llm_model or config_llm_model
        self.llm_temperature = (
            eval_llm_temperature if eval_llm_temperature is not None else config_llm_temp
        )
        self.llm_client = build_llm_client() if self.use_llm else None
        if self.use_llm and self.llm_client is None:
            self.use_llm = False

        self.store_predicted = store_predicted
        if truncate_predicted is None:
            truncate_predicted = True
        self.truncate_predicted = truncate_predicted

        if predicted_max_chars is None:
            predicted_max_chars = 200
        self.predicted_max_chars = predicted_max_chars
        if self.truncate_predicted and self.predicted_max_chars <= 0:
            self.predicted_max_chars = 200

        if self.eval_config is not None:
            # Initialize evaluators for each dataset type present
            self._evaluators = {
                "bizfinbench": BizFinBenchEvaluator(
                    use_llm=self.use_llm,
                    llm_client=self.llm_client,
                    llm_model=self.llm_model,
                    llm_temperature=self.llm_temperature,
                ),
                "public_csv": PublicCsvEvaluator(
                    use_llm=self.use_llm,
                    llm_client=self.llm_client,
                    llm_model=self.llm_model,
                    llm_temperature=self.llm_temperature,
                ),
                "gdpval": GDPValEvaluator(
                    use_llm=self.use_llm,
                    llm_client=self.llm_client,
                    llm_model=self.llm_model,
                    llm_temperature=self.llm_temperature,
                ),
                "synthetic": self.evaluator,  # Use ComprehensiveEvaluator
                "options": None,  # Options use OptionsEvaluator initialized per-task
                "crypto": None,  # Crypto uses CryptoTradingEvaluator initialized per-scenario
            }
            
        elif dataset_type == "bizfinbench" and dataset_path:
            # Legacy single BizFinBench dataset
            self.dataset_provider = BizFinBenchProvider(
                base_path=dataset_path,
                task_type=task_type or "event_logic_reasoning",
                language=language,
                limit=limit,
            )
            self.dataset_evaluator = BizFinBenchEvaluator(
                use_llm=self.use_llm,
                llm_client=self.llm_client,
                llm_model=self.llm_model,
                llm_temperature=self.llm_temperature,
            )
            self._examples = self.dataset_provider.load()
            
        elif dataset_type == "public_csv" and dataset_path:
            # Legacy single public.csv dataset
            self.dataset_provider = CsvFinanceDatasetProvider(path=dataset_path)
            self.dataset_evaluator = PublicCsvEvaluator(
                use_llm=self.use_llm,
                llm_client=self.llm_client,
                llm_model=self.llm_model,
                llm_temperature=self.llm_temperature,
            )
            examples = self.dataset_provider.load()
            self._examples = examples[:limit] if limit else examples

    async def run_eval(self, request: Any, updater: TaskUpdater) -> None:
        """
        Run evaluation with EvalRequest from agentbeats-client.

        Args:
            request: EvalRequest with participants and config
            updater: TaskUpdater for reporting progress and results
        """
        # Get participant URL from request (first participant)
        participants = request.participants
        if not participants:
            raise ValueError("No participants provided in request")

        purple_agent_url = str(list(participants.values())[0])
        config = request.config or {}

        # Get config values with defaults
        num_tasks = int(config.get("num_tasks", os.environ.get("EVAL_NUM_TASKS", "10")))
        conduct_debate = config.get("conduct_debate", os.environ.get("EVAL_CONDUCT_DEBATE", "false"))
        if isinstance(conduct_debate, str):
            conduct_debate = conduct_debate.lower() == "true"

        await self._run_evaluation(
            purple_agent_url=purple_agent_url,
            num_tasks=num_tasks,
            conduct_debate=conduct_debate,
            updater=updater,
        )

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Run the FAB++ evaluation assessment (legacy method).

        Args:
            message: The incoming A2A message (plain text trigger)
            updater: TaskUpdater for reporting progress and results
        """
        input_text = get_message_text(message)

        # Get configuration from environment variables (set by docker-compose/scenario)
        purple_agent_url = os.environ.get("PURPLE_AGENT_URL", "http://purple_agent:9009")
        num_tasks = int(os.environ.get("EVAL_NUM_TASKS", "10"))
        conduct_debate = os.environ.get("EVAL_CONDUCT_DEBATE", "false").lower() == "true"

        await self._run_evaluation(
            purple_agent_url=purple_agent_url,
            num_tasks=num_tasks,
            conduct_debate=conduct_debate,
            updater=updater,
        )

    async def _run_evaluation(
        self,
        purple_agent_url: str,
        num_tasks: int,
        conduct_debate: bool,
        updater: TaskUpdater,
    ) -> None:
        """
        Internal method to run the actual evaluation.

        Args:
            purple_agent_url: URL of the purple agent to evaluate
            num_tasks: Number of tasks to evaluate
            conduct_debate: Whether to conduct adversarial debate
            updater: TaskUpdater for reporting progress and results
        """
        ticker = os.environ.get("EVAL_TICKER", "NVDA")

        # Report starting
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting FAB++ evaluation for {ticker}...")
        )

        try:
            # Generate evaluation task(s)
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Generating evaluation tasks...")
            )
            
            # Get simulation date from environment or use current date
            from datetime import datetime
            simulation_date_str = os.environ.get("SIMULATION_DATE")
            if simulation_date_str:
                simulation_date = datetime.fromisoformat(simulation_date_str)
            else:
                simulation_date = datetime.now()
            
            # Use synthetic questions if available, otherwise generate tasks
            # Priority: config-based > legacy single dataset > synthetic > dynamic generation
            if self._loaded_examples is not None:
                # Config-based multi-dataset evaluation
                summary = self.dataset_loader.summary()
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"Using {summary['total']} examples from {len(summary['by_dataset'])} datasets..."
                    )
                )
                all_results = await self._evaluate_with_config(
                    purple_agent_url=purple_agent_url,
                    num_tasks=num_tasks,
                    conduct_debate=conduct_debate,
                    updater=updater,
                )

                # Use unified scoring system
                scorer = UnifiedScorer()
                normalized_results = []

                for r in all_results:
                    if "error" in r:
                        continue

                    dataset_type = r.get("dataset_type", "unknown")
                    raw_score = r.get("score", 0.0)
                    is_correct = r.get("is_correct", False)

                    # Extract sub-scores for options
                    sub_scores = {}
                    if dataset_type == "options":
                        sub_scores = {
                            "pnl_accuracy": r.get("pnl_accuracy", 0),
                            "greeks_accuracy": r.get("greeks_accuracy", 0),
                            "strategy_quality": r.get("strategy_quality", 0),
                            "risk_management": r.get("risk_management", 0),
                        }
                    elif dataset_type == "crypto":
                        sub_scores = {
                            "baseline": r.get("baseline_score", 0),
                            "noisy": r.get("noisy_score", 0),
                            "adversarial": r.get("adversarial_score", 0),
                            "meta": r.get("meta_score", 0),
                        }
                    elif dataset_type == "gdpval":
                        sub_scores = {
                            "completion": r.get("completion", 0),
                            "accuracy": r.get("accuracy", 0),
                            "format": r.get("format", 0),
                            "professionalism": r.get("professionalism", 0),
                        }

                    normalized = scorer.create_normalized_result(
                        task_id=r.get("example_id", ""),
                        dataset_type=dataset_type,
                        raw_score=raw_score,
                        is_correct=is_correct,
                        feedback=r.get("feedback", ""),
                        sub_scores=sub_scores,
                    )
                    if normalized:
                        normalized_results.append(normalized)

                # Compute unified result
                unified_result = scorer.compute_unified_result(
                    task_results=normalized_results,
                    purple_agent_url=purple_agent_url,
                    conduct_debate=conduct_debate,
                )
                if set(summary["by_dataset"].keys()) == {"crypto"}:
                    unified_result.benchmark = "AgentBusters Crypto Trading Benchmark"
                    unified_result.version = "1.0.0"

                # Convert to dict for serialization
                assessment_result = unified_result.to_dict()

                # Add config summary for compatibility
                assessment_result["config_summary"] = summary
                assessment_result["results"] = all_results  # Keep detailed results

                # Save AgentBeats-compliant results
                participant_id = os.environ.get("AGENTBEATS_PURPLE_AGENT_ID", "")
                participant_name = os.environ.get("PARTICIPANT_NAME", "purple_agent")
                scenario_id = os.environ.get("AGENTBEATS_SCENARIO_ID", "")
                green_agent_id = os.environ.get("AGENTBEATS_GREEN_AGENT_ID", "")

                try:
                    results_path, leaderboard_path = format_and_save_results(
                        participant_id=participant_id,
                        participant_name=participant_name,
                        evaluation_results=assessment_result,
                        by_dataset=None,  # Unified result handles this differently
                        scenario_id=scenario_id,
                        green_agent_id=green_agent_id,
                        results_dir="results",
                    )
                    import structlog
                    logger = structlog.get_logger()
                    logger.info("agentbeats_results_saved", results_path=str(results_path), leaderboard_path=str(leaderboard_path))
                except Exception as e:
                    import structlog
                    logger = structlog.get_logger()
                    logger.warning("agentbeats_results_save_failed", error=str(e))

                # Report results as artifact
                overall = unified_result.overall_score
                section_summary = "\n".join(
                    f"  {name}: {ss.score:.1f}/100 (weight: {ss.weight:.0%}, {ss.task_count} tasks)"
                    for name, ss in unified_result.section_scores.items()
                )
                await updater.add_artifact(
                    parts=[
                        Part(root=TextPart(text=f"FAB++ Unified Evaluation Complete\n\nOverall Score: {overall.score:.1f}/100 (Grade: {overall.grade})\n\nSection Scores:\n{section_summary}")),
                        Part(root=DataPart(data=assessment_result)),
                    ],
                    name="evaluation_result",
                )
                return
            
            elif self._examples and self.dataset_type in ("bizfinbench", "public_csv"):
                # Legacy: Use single dataset examples directly
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Using {len(self._examples)} {self.dataset_type} examples for evaluation...")
                )
                # For dataset-based evaluation, we'll use a different flow
                all_results = await self._evaluate_with_dataset(
                    purple_agent_url=purple_agent_url,
                    num_tasks=num_tasks,
                    conduct_debate=conduct_debate,
                    updater=updater,
                )
                
                # Calculate aggregate metrics
                valid_results = [r for r in all_results if "error" not in r]
                avg_score = sum(r.get("score", 0) for r in valid_results) / len(valid_results) if valid_results else 0.0
                accuracy = sum(1 for r in valid_results if r.get("is_correct", False)) / len(valid_results) if valid_results else 0.0
                
                # Create assessment result
                assessment_result = {
                    "benchmark": f"FAB++ {self.dataset_type}",
                    "version": "1.1.0",
                    "purple_agent": purple_agent_url,
                    "dataset_type": self.dataset_type,
                    "task_type": self.task_type,
                    "language": self.language,
                    "num_examples": len(self._examples),
                    "num_evaluated": len(all_results),
                    "num_successful": len(valid_results),
                    "average_score": round(avg_score, 4),
                    "accuracy": round(accuracy, 4),
                    "results": all_results,
                }
                
                # Report results as artifact
                await updater.add_artifact(
                    parts=[
                        Part(root=TextPart(text=f"FAB++ {self.dataset_type} Evaluation Complete\\n\\nAverage Score: {avg_score:.4f}\\nAccuracy: {accuracy:.2%}")),
                        Part(root=DataPart(data=assessment_result)),
                    ],
                    name="evaluation_result",
                )
                return
            
            elif self.synthetic_questions:
                tasks = self._convert_synthetic_to_tasks(num_tasks)
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Using {len(tasks)} synthetic questions for evaluation...")
                )
            else:
                tasks = await self.task_generator.generate_task_batch(
                    count=num_tasks,
                    simulation_date=simulation_date,
                )
            
            if not tasks:
                # Create a default task if generation fails
                from datetime import datetime
                from cio_agent.models import GroundTruth, FinancialData, TaskRubric

                # Convert task_category string to enum
                try:
                    default_category = TaskCategory(task_category)
                except ValueError:
                    default_category = TaskCategory.BEAT_OR_MISS

                tasks = [FABTask(
                    question_id=f"fab_{ticker}_eval",
                    category=default_category,
                    question=f"Did {ticker} beat or miss analyst expectations in the most recent quarter?",
                    ticker=ticker,
                    fiscal_year=2026,
                    simulation_date=datetime.now(),
                    ground_truth=GroundTruth(
                        macro_thesis="Evaluate earnings performance",
                        key_themes=["revenue", "earnings", "guidance"],
                        financials=FinancialData(),
                        expected_recommendation="Evaluate",
                    ),
                    difficulty=TaskDifficulty.MEDIUM,
                    rubric=TaskRubric(
                        criteria=["Accuracy", "Analysis depth", "Recommendation quality"],
                        mandatory_elements=["beat/miss determination"],
                    ),
                )]

            all_results = []
            
            for i, task in enumerate(tasks):
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Evaluating task {i+1}/{len(tasks)}: {task.question_id}")
                )
                
                # Send task to Purple Agent
                task_message = json.dumps({
                    "question": task.question,
                    "ticker": task.ticker,
                    "fiscal_year": task.fiscal_year,
                    "category": task.category.value,
                }, ensure_ascii=False)
                
                try:
                    response = await self.messenger.talk_to_agent(
                        message=task_message,
                        url=purple_agent_url,
                        new_conversation=True,
                        timeout=300,
                    )
                    
                    # Parse agent response
                    from cio_agent.models import AgentResponse, FinancialData as FD
                    agent_response = AgentResponse(
                        agent_id="purple_agent",
                        task_id=task.question_id,
                        analysis=response,
                        recommendation=self._extract_recommendation(response),
                        extracted_financials=FD(),  # Would be parsed from response
                        tool_calls=[],
                        code_executions=[],
                        execution_time_seconds=0.0,
                    )
                    
                    # Conduct debate if enabled
                    agent_rebuttal = None
                    if conduct_debate:
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message("Conducting adversarial debate...")
                        )
                        
                        counter_arg = f"Challenge: What are the key risks to your {ticker} analysis?"
                        rebuttal_response = await self.messenger.talk_to_agent(
                            message=counter_arg,
                            url=purple_agent_url,
                            new_conversation=False,
                        )
                        
                        from cio_agent.models import DebateRebuttal
                        agent_rebuttal = DebateRebuttal(
                            agent_id="purple_agent",
                            task_id=task.question_id,
                            defense=rebuttal_response,
                        )
                    
                    # Evaluate response
                    result = await self.evaluator.evaluate_response(
                        task=task,
                        agent_response=agent_response,
                        agent_rebuttal=agent_rebuttal,
                    )
                    
                    all_results.append({
                        "task_id": task.question_id,
                        "alpha_score": result.alpha_score.score,
                        "role_score": result.role_score.total,
                        "debate_multiplier": result.debate_result.debate_multiplier,
                    })
                    
                except Exception as e:
                    all_results.append({
                        "task_id": task.question_id,
                        "error": str(e),
                        "alpha_score": 0.0,
                    })

            # Calculate aggregate metrics
            valid_results = [r for r in all_results if "error" not in r]
            avg_alpha = sum(r["alpha_score"] for r in valid_results) / len(valid_results) if valid_results else 0.0
            
            # Create assessment result
            assessment_result = {
                "benchmark": "FAB++ Finance Agent Benchmark",
                "version": "1.0.0",
                "purple_agent": purple_agent_url,
                "ticker": ticker,
                "num_tasks": len(tasks),
                "num_successful": len(valid_results),
                "average_alpha_score": round(avg_alpha, 2),
                "results": all_results,
            }
            
            # Report results as artifact
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=f"FAB++ Evaluation Complete\n\nAverage Alpha Score: {avg_alpha:.2f}")),
                    Part(root=DataPart(data=assessment_result)),
                ],
                name="evaluation_result",
            )

        except Exception as e:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Evaluation failed: {str(e)}")
            )
            raise

    def _extract_recommendation(self, response: str) -> str:
        """Extract recommendation from agent response."""
        response_lower = response.lower()
        if "beat" in response_lower:
            return "Beat"
        elif "miss" in response_lower:
            return "Miss"
        elif "buy" in response_lower:
            return "Buy"
        elif "sell" in response_lower:
            return "Sell"
        elif "hold" in response_lower:
            return "Hold"
        return "Unknown"

    def _format_predicted(self, response: str) -> str:
        if not self.store_predicted:
            return ""
        if not self.truncate_predicted or self.predicted_max_chars <= 0:
            return response
        if len(response) <= self.predicted_max_chars:
            return response
        return response[: self.predicted_max_chars] + "..."

    def _convert_synthetic_to_tasks(self, num_tasks: int) -> list[FABTask]:
        """
        Convert synthetic question dicts to FABTask objects.
        
        Args:
            num_tasks: Maximum number of tasks to return
            
        Returns:
            List of FABTask objects
        """
        from datetime import datetime
        
        tasks = []
        questions_to_use = self.synthetic_questions[:num_tasks]
        
        for sq in questions_to_use:
            # Handle category enum
            category_value = sq.get("category", "Quantitative Retrieval")
            try:
                category = TaskCategory(category_value)
            except ValueError:
                category = TaskCategory.QUANTITATIVE_RETRIEVAL
            
            # Handle difficulty enum
            difficulty_value = sq.get("difficulty", "medium")
            try:
                difficulty = TaskDifficulty(difficulty_value)
            except ValueError:
                difficulty = TaskDifficulty.MEDIUM
            
            # Build ground truth with required fields
            ground_truth = GroundTruth(
                macro_thesis=str(sq.get("ground_truth_formatted", "Evaluate the analysis")),
                key_themes=sq.get("calculation_steps", []),
                expected_recommendation=str(sq.get("ground_truth_formatted", "")),
                financials=FinancialData(),
            )
            
            # Build rubric from components
            rubric_data = sq.get("rubric", {})
            rubric_components = rubric_data.get("components", [])
            rubric = TaskRubric(
                criteria=[c.get("description", "") for c in rubric_components],
                max_score=rubric_data.get("max_score", 100),
            )
            
            task = FABTask(
                question_id=sq.get("question_id", f"SYN_{len(tasks):04d}"),
                category=category,
                difficulty=difficulty,
                question=sq.get("question", ""),
                ticker=sq.get("ticker", "AAPL"),
                fiscal_year=sq.get("fiscal_year", 2024),
                simulation_date=datetime.now(),
                ground_truth=ground_truth,
                rubric=rubric,
                requires_code_execution=sq.get("requires_code_execution", False),
            )
            tasks.append(task)
        
        return tasks

    async def _evaluate_with_dataset(
        self,
        purple_agent_url: str,
        num_tasks: int,
        conduct_debate: bool,
        updater: TaskUpdater,
    ) -> List[dict]:
        """
        Evaluate Purple Agent using dataset examples and dataset-specific evaluator.
        
        Args:
            purple_agent_url: URL of the Purple Agent to evaluate
            num_tasks: Maximum number of examples to evaluate
            conduct_debate: Whether to conduct adversarial debate
            updater: TaskUpdater for progress reporting
            
        Returns:
            List of evaluation results
        """
        all_results = []
        examples_to_eval = self._examples[:num_tasks] if num_tasks else self._examples
        
        for i, example in enumerate(examples_to_eval):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Evaluating example {i+1}/{len(examples_to_eval)}: {example.example_id}")
            )
            
            try:
                # Send question to Purple Agent
                response = await self.messenger.talk_to_agent(
                    message=example.question,
                    url=purple_agent_url,
                    new_conversation=True,
                    timeout=300,
                )
                predicted_text = self._format_predicted(response)
                
                # Use dataset-specific evaluator
                if self.dataset_type == "bizfinbench":
                    eval_result = self.dataset_evaluator.evaluate(
                        predicted=response,
                        expected=example.answer,
                        task_type=self.task_type,
                        question=example.question,
                    )
                    result = {
                        "example_id": example.example_id,
                        "task_type": self.task_type,
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "expected": example.answer[:100] + "..." if len(example.answer) > 100 else example.answer,
                        "predicted": predicted_text,
                        "score": eval_result.score,
                        "is_correct": eval_result.is_correct,
                        "feedback": eval_result.feedback,
                    }
                    if eval_result.details:
                        for key in ("llm_used", "llm_failure", "llm_raw_output"):
                            if key in eval_result.details:
                                result[key] = eval_result.details.get(key)
                    result["llm_used"] = eval_result.details.get("llm_used", False) if eval_result.details else False
                    result["sub_scores"] = {}
                    
                elif self.dataset_type == "public_csv":
                    # Build rubric from example
                    rubric_list = []
                    if hasattr(example, 'rubric') and example.rubric:
                        for criterion in getattr(example.rubric, 'criteria', []):
                            rubric_list.append({"operator": "correctness", "criteria": criterion})
                        for penalty in getattr(example.rubric, 'penalty_conditions', []):
                            rubric_list.append({"operator": "contradiction", "criteria": penalty})
                    
                    eval_result = self.dataset_evaluator.evaluate(
                        predicted=response,
                        expected=example.answer,
                        rubric=rubric_list if rubric_list else None,
                        question=example.question,
                    )
                    result = {
                        "example_id": example.example_id,
                        "category": example.category.value if hasattr(example.category, 'value') else str(example.category),
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "expected": example.answer[:100] + "..." if len(example.answer) > 100 else example.answer,
                        "predicted": predicted_text,
                        "score": eval_result.score,
                        "is_correct": eval_result.score >= 0.8,  # Threshold for correctness
                        "correct_count": eval_result.correct_count,
                        "total_count": eval_result.total_count,
                        "feedback": eval_result.feedback,
                    }
                    if eval_result.details:
                        for key in (
                            "llm_used",
                            "llm_failure",
                            "llm_raw_output",
                            "llm_partial",
                            "llm_item_count_expected",
                            "llm_item_count_actual",
                        ):
                            if key in eval_result.details:
                                result[key] = eval_result.details.get(key)
                    result["llm_used"] = eval_result.details.get("llm_used", False) if eval_result.details else False
                    result["sub_scores"] = {}
                else:
                    result = {
                        "example_id": example.example_id,
                        "error": f"Unknown dataset type: {self.dataset_type}",
                    }
                
                # Optional debate (simplified for dataset-based evaluation)
                if conduct_debate and eval_result.score > 0:
                    challenge = f"Challenge your analysis. What risks or uncertainties did you consider?"
                    try:
                        rebuttal = await self.messenger.talk_to_agent(
                            message=challenge,
                            url=purple_agent_url,
                            new_conversation=False,
                            timeout=60,
                        )
                        result["rebuttal_received"] = True
                        result["rebuttal_preview"] = rebuttal[:100] + "..." if len(rebuttal) > 100 else rebuttal
                    except Exception:
                        result["rebuttal_received"] = False
                
                all_results.append(result)
                
            except Exception as e:
                all_results.append({
                    "example_id": example.example_id,
                    "error": str(e),
                    "score": 0.0,
                    "is_correct": False,
                })
        
        return all_results



    async def _evaluate_with_config(
        self,
        purple_agent_url: str,
        num_tasks: int,
        conduct_debate: bool,
        updater: TaskUpdater,
    ) -> List[dict]:
        """
        Evaluate Purple Agent using config-based multi-dataset loader.
        
        Args:
            purple_agent_url: URL of the Purple Agent to evaluate
            num_tasks: Maximum number of examples to evaluate
            conduct_debate: Whether to conduct adversarial debate
            updater: TaskUpdater for progress reporting
            
        Returns:
            List of evaluation results
        """
        all_results = []
        crypto_evaluator = None
        examples_to_eval = self._loaded_examples[:num_tasks] if num_tasks else self._loaded_examples
        
        for i, example in enumerate(examples_to_eval):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[{i+1}/{len(examples_to_eval)}] Evaluating {example.dataset_type}: {example.example_id}"
                )
            )
            
            try:
                response = ""
                if example.dataset_type != "crypto":
                    # Send question to Purple Agent
                    response = await self.messenger.talk_to_agent(
                        message=example.question,
                        url=purple_agent_url,
                        new_conversation=True,
                        timeout=self.eval_config.timeout_seconds if self.eval_config else 300,
                    )
                predicted_text = self._format_predicted(response)
                
                # Get appropriate evaluator (options handled specially below)
                evaluator = self._evaluators.get(example.dataset_type)
                if not evaluator and example.dataset_type not in ("options", "crypto"):
                    all_results.append({
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "error": f"No evaluator for dataset type: {example.dataset_type}",
                        "score": 0.0,
                        "is_correct": False,
                    })
                    continue

                # Evaluate based on dataset type
                if example.dataset_type == "bizfinbench":
                    eval_result = evaluator.evaluate(
                        predicted=response,
                        expected=example.answer,
                        task_type=example.task_type,
                        question=example.question,
                    )
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "task_type": example.task_type,
                        "language": example.language,
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "expected": example.answer[:100] + "..." if len(example.answer) > 100 else example.answer,
                        "predicted": predicted_text,
                        "score": eval_result.score,
                        "is_correct": eval_result.is_correct,
                        "feedback": eval_result.feedback,
                    }
                    if eval_result.details:
                        for key in ("llm_used", "llm_failure", "llm_raw_output"):
                            if key in eval_result.details:
                                result[key] = eval_result.details.get(key)
                    
                elif example.dataset_type == "public_csv":
                    # Build rubric from metadata if available
                    rubric_list = []
                    rubric_data = example.metadata.get("rubric")
                    if rubric_data:
                        if hasattr(rubric_data, 'criteria'):
                            for criterion in rubric_data.criteria:
                                rubric_list.append({"operator": "correctness", "criteria": criterion})
                        if hasattr(rubric_data, 'penalty_conditions'):
                            for penalty in rubric_data.penalty_conditions:
                                rubric_list.append({"operator": "contradiction", "criteria": penalty})
                    
                    eval_result = evaluator.evaluate(
                        predicted=response,
                        expected=example.answer,
                        rubric=rubric_list if rubric_list else None,
                        question=example.question,
                    )
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "category": example.category,
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "expected": example.answer[:100] + "..." if len(example.answer) > 100 else example.answer,
                        "predicted": predicted_text,
                        "score": eval_result.score,
                        "is_correct": eval_result.score >= 0.8,
                        "correct_count": eval_result.correct_count,
                        "total_count": eval_result.total_count,
                        "feedback": eval_result.feedback,
                    }
                    if eval_result.details:
                        for key in (
                            "llm_used",
                            "llm_failure",
                            "llm_raw_output",
                            "llm_partial",
                            "llm_item_count_expected",
                            "llm_item_count_actual",
                        ):
                            if key in eval_result.details:
                                result[key] = eval_result.details.get(key)
                    
                elif example.dataset_type == "synthetic":
                    # Synthetic questions use recommendation extraction
                    extracted = self._extract_recommendation(response)
                    expected = self._extract_recommendation(example.answer) if example.answer else ""

                    is_correct = extracted.lower() == expected.lower() if expected else False

                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "category": example.category,
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "expected": expected,
                        "predicted": extracted,
                        "predicted_full": predicted_text,
                        "score": 1.0 if is_correct else 0.0,
                        "is_correct": is_correct,
                        "feedback": f"Extracted: {extracted}, Expected: {expected}",
                    }
                    result["llm_used"] = False
                    result["sub_scores"] = {}
                    eval_result = type('obj', (object,), {'score': result['score']})()

                elif example.dataset_type == "gdpval":
                    # GDPVal: Open-ended professional tasks (LLM-as-judge)
                    eval_result = evaluator.evaluate(
                        predicted=response,
                        expected="",  # GDPVal has no ground truth
                        task_prompt=example.question,
                        occupation=example.task_type,  # task_type stores occupation
                        sector=example.category,  # category stores sector
                        reference_files=example.metadata.get("reference_files", []),
                        question=example.question,
                    )
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "occupation": example.task_type,
                        "sector": example.category,
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "predicted": predicted_text,
                        "score": eval_result.score,
                        "is_correct": eval_result.score >= 0.7,  # 70% threshold
                        "feedback": eval_result.feedback,
                        "has_reference_files": example.metadata.get("has_reference_files", False),
                    }
                    # Add detailed scores if available
                    sub_scores: dict[str, float] = {}
                    if eval_result.details:
                        for key in ("completion", "accuracy", "format", "professionalism", "llm_used"):
                            if key in eval_result.details:
                                result[key] = eval_result.details[key]
                                if key in ("completion", "accuracy", "format", "professionalism"):
                                    sub_scores[key] = float(eval_result.details[key])
                    result["llm_used"] = eval_result.details.get("llm_used", False) if eval_result.details else False
                    result["sub_scores"] = sub_scores

                elif example.dataset_type == "options":
                    # Options Alpha Challenge evaluation
                    from cio_agent.models import AgentResponse
                    from datetime import datetime, timezone

                    # Map string category to TaskCategory enum
                    category_map = {
                        "Options Pricing": TaskCategory.OPTIONS_PRICING,
                        "Greeks Analysis": TaskCategory.GREEKS_ANALYSIS,
                        "Strategy Construction": TaskCategory.STRATEGY_CONSTRUCTION,
                        "Volatility Trading": TaskCategory.VOLATILITY_TRADING,
                        "P&L Attribution": TaskCategory.PNL_ATTRIBUTION,
                        "Risk Management": TaskCategory.RISK_MANAGEMENT,
                        "Copy Trading": TaskCategory.COPY_TRADING,
                        "Race to 10M": TaskCategory.RACE_TO_10M,
                        "Strategy Defense": TaskCategory.STRATEGY_DEFENSE,
                    }
                    task_category = category_map.get(example.category, TaskCategory.OPTIONS_PRICING)

                    # Extract ticker from metadata if available
                    ticker = example.metadata.get("ticker", "SPY")

                    # Create a FABTask for the evaluator
                    fab_task = FABTask(
                        question_id=example.example_id,
                        category=task_category,
                        difficulty=TaskDifficulty.MEDIUM,
                        question=example.question,
                        ticker=ticker,
                        fiscal_year=2025,
                        simulation_date=datetime.now(timezone.utc),
                        ground_truth=GroundTruth(
                            macro_thesis=example.answer,
                            key_themes=[example.category],
                        ),
                        rubric=TaskRubric(criteria=[], penalty_conditions=[]),
                    )

                    # Create AgentResponse
                    agent_response = AgentResponse(
                        agent_id="purple_agent",
                        task_id=example.example_id,
                        analysis=response,
                        recommendation=self._extract_recommendation(response),
                    )

                    # Initialize OptionsEvaluator with the task
                    options_evaluator = OptionsEvaluator(task=fab_task)
                    options_score = await options_evaluator.score(agent_response)

                    # Options scores are already on 0-100 scale - don't normalize here
                    # The unified scorer handles 0-100 scale for options
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "category": example.category,
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "expected": example.answer[:100] + "..." if len(example.answer) > 100 else example.answer,
                        "predicted": predicted_text,
                        "score": options_score.score,  # Keep as 0-100 scale
                        "is_correct": options_score.score >= 70,  # 70/100 threshold
                        "pnl_accuracy": options_score.pnl_accuracy,
                        "greeks_accuracy": options_score.greeks_accuracy,
                        "strategy_quality": options_score.strategy_quality,
                        "risk_management": options_score.risk_management,
                        "feedback": options_score.feedback,
                    }
                    result["llm_used"] = False
                    result["sub_scores"] = {
                        "pnl_accuracy": options_score.pnl_accuracy,
                        "greeks_accuracy": options_score.greeks_accuracy,
                        "strategy_quality": options_score.strategy_quality,
                        "risk_management": options_score.risk_management,
                    }
                    eval_result = type('obj', (object,), {'score': options_score.score})()

                elif example.dataset_type == "crypto":
                    if crypto_evaluator is None:
                        crypto_evaluator = CryptoTradingEvaluator(
                            messenger=self.messenger,
                            timeout_seconds=self.eval_config.timeout_seconds if self.eval_config else 300,
                        )

                    scenario_meta = example.metadata or {}
                    scenario_seed_base = os.environ.get("AGENTBEATS_PURPLE_AGENT_ID") or purple_agent_url
                    scenario_seed = scenario_meta.get("seed")
                    if scenario_seed is None:
                        scenario_seed = stable_seed(scenario_seed_base, example.example_id)

                    crypto_result = await crypto_evaluator.evaluate_scenario(
                        scenario_meta=scenario_meta,
                        purple_agent_url=purple_agent_url,
                        seed=scenario_seed,
                    )

                    if "error" in crypto_result:
                        result = {
                            "example_id": example.example_id,
                            "dataset_type": example.dataset_type,
                            "error": crypto_result["error"],
                            "score": 0.0,
                            "is_correct": False,
                            "llm_used": False,
                            "sub_scores": {},
                        }
                        eval_result = type('obj', (object,), {'score': 0.0})()
                    else:
                        result = {
                            "example_id": example.example_id,
                            "dataset_type": example.dataset_type,
                            "scenario_id": scenario_meta.get("scenario_id", example.example_id),
                            "scenario_name": scenario_meta.get("name", example.example_id),
                            "score": crypto_result["final_score"],
                            "baseline_score": crypto_result["baseline"]["score"],
                            "noisy_score": crypto_result["noisy"]["score"],
                            "adversarial_score": crypto_result["adversarial"]["score"],
                            "meta_score": crypto_result["meta"]["score"],
                            "grade": crypto_result["grade"],
                            "random_seed": crypto_result["random_seed"],
                            "metrics": {
                                "baseline": crypto_result["baseline"]["metrics"],
                                "noisy": crypto_result["noisy"]["metrics"],
                                "adversarial": crypto_result["adversarial"]["metrics"],
                                "meta": crypto_result["meta"],
                            },
                            "events": crypto_result.get("events", []),
                            "llm_used": False,
                            "sub_scores": {
                                "baseline": crypto_result["baseline"]["score"],
                                "noisy": crypto_result["noisy"]["score"],
                                "adversarial": crypto_result["adversarial"]["score"],
                                "meta": crypto_result["meta"]["score"],
                            },
                        }
                        result["is_correct"] = crypto_result["final_score"] >= 70
                        result["feedback"] = (
                            f"Final score {crypto_result['final_score']:.2f} "
                            f"(grade {crypto_result['grade']})"
                        )
                        eval_result = type('obj', (object,), {'score': crypto_result["final_score"]})()

                else:
                    # Generic handling for unknown types
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "predicted": predicted_text,
                        "score": 0.0,  # No evaluator, no score
                        "is_correct": False,
                        "feedback": "No evaluator configured for this dataset type",
                        "llm_used": False,
                        "sub_scores": {},
                    }
                
                # Optional debate
                if conduct_debate and eval_result.score > 0 and example.dataset_type != "crypto":
                    try:
                        rebuttal = await self.messenger.talk_to_agent(
                            message="Challenge your analysis. What risks or uncertainties did you consider?",
                            url=purple_agent_url,
                            new_conversation=False,
                            timeout=60,
                        )
                        result["rebuttal_received"] = True
                        result["rebuttal_preview"] = rebuttal[:100] + "..." if len(rebuttal) > 100 else rebuttal
                    except Exception:
                        result["rebuttal_received"] = False
                
                all_results.append(result)
                
            except Exception as e:
                all_results.append({
                    "example_id": example.example_id,
                    "dataset_type": example.dataset_type,
                    "error": str(e),
                    "score": 0.0,
                    "is_correct": False,
                })
        
        return all_results



class EvalRequest(BaseModel):
    """Evaluation request payload."""

    participants: dict[str, str]
    config: dict[str, Any] = {}


