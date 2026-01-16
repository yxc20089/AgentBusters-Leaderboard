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
"""

import json
from pathlib import Path
from typing import Any, Optional, List, Union
from pydantic import BaseModel, HttpUrl, ValidationError
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

# Dataset providers (for legacy single-dataset mode)
from cio_agent.datasets import BizFinBenchProvider, CsvFinanceDatasetProvider

# Dataset-specific evaluators
from evaluators import BizFinBenchEvaluator, PublicCsvEvaluator, OptionsEvaluator


class EvalRequest(BaseModel):
    """
    Request format sent by the AgentBeats platform to green agents.
    
    The platform sends this JSON structure when initiating an assessment.
    """
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


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

    def __init__(
        self,
        eval_config: Optional[Union[EvaluationConfig, str, Path]] = None,
        synthetic_questions: Optional[List[dict]] = None,
        dataset_type: str = "synthetic",
        dataset_path: Optional[str] = None,
        task_type: Optional[str] = None,
        language: str = "en",
        limit: Optional[int] = None,
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
            
            # Initialize evaluators for each dataset type present
            self._evaluators = {
                "bizfinbench": BizFinBenchEvaluator(),
                "public_csv": PublicCsvEvaluator(),
                "synthetic": self.evaluator,  # Use ComprehensiveEvaluator
                "options": None,  # Options use OptionsEvaluator initialized per-task
            }
            
        elif dataset_type == "bizfinbench" and dataset_path:
            # Legacy single BizFinBench dataset
            self.dataset_provider = BizFinBenchProvider(
                base_path=dataset_path,
                task_type=task_type or "event_logic_reasoning",
                language=language,
                limit=limit,
            )
            self.dataset_evaluator = BizFinBenchEvaluator()
            self._examples = self.dataset_provider.load()
            
        elif dataset_type == "public_csv" and dataset_path:
            # Legacy single public.csv dataset
            self.dataset_provider = CsvFinanceDatasetProvider(path=dataset_path)
            self.dataset_evaluator = PublicCsvEvaluator()
            examples = self.dataset_provider.load()
            self._examples = examples[:limit] if limit else examples


    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate the assessment request."""
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Run the FAB++ evaluation assessment.

        Args:
            message: The incoming A2A message containing the EvalRequest
            updater: TaskUpdater for reporting progress and results
        """
        input_text = get_message_text(message)

        # Parse and validate the assessment request
        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        # Extract configuration
        purple_agent_url = str(request.participants["purple_agent"])
        ticker = request.config.get("ticker", "NVDA")
        task_category = request.config.get("task_category", "beat_or_miss")
        num_tasks = request.config.get("num_tasks", 1)
        conduct_debate = request.config.get("conduct_debate", True)

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
            
            # Get simulation date from config or use current date
            from datetime import datetime
            simulation_date_str = request.config.get("simulation_date")
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
                
                # Calculate aggregate metrics
                valid_results = [r for r in all_results if "error" not in r]
                avg_score = sum(r.get("score", 0) for r in valid_results) / len(valid_results) if valid_results else 0.0
                accuracy = sum(1 for r in valid_results if r.get("is_correct", False)) / len(valid_results) if valid_results else 0.0
                
                # Group by dataset
                by_dataset = {}
                for r in valid_results:
                    ds = r.get("dataset_type", "unknown")
                    if ds not in by_dataset:
                        by_dataset[ds] = {"scores": [], "correct": 0}
                    by_dataset[ds]["scores"].append(r.get("score", 0))
                    if r.get("is_correct", False):
                        by_dataset[ds]["correct"] += 1
                
                # Create assessment result
                assessment_result = {
                    "benchmark": f"FAB++ {self.eval_config.name}",
                    "version": self.eval_config.version,
                    "purple_agent": purple_agent_url,
                    "config_summary": summary,
                    "num_evaluated": len(all_results),
                    "num_successful": len(valid_results),
                    "average_score": round(avg_score, 4),
                    "accuracy": round(accuracy, 4),
                    "by_dataset": {
                        ds: {
                            "count": len(data["scores"]),
                            "mean_score": round(sum(data["scores"]) / len(data["scores"]), 4) if data["scores"] else 0,
                            "accuracy": round(data["correct"] / len(data["scores"]), 4) if data["scores"] else 0,
                        }
                        for ds, data in by_dataset.items()
                    },
                    "results": all_results,
                }
                
                # Report results as artifact
                await updater.add_artifact(
                    parts=[
                        Part(root=TextPart(text=f"FAB++ Multi-Dataset Evaluation Complete\n\nAverage Score: {avg_score:.4f}\nAccuracy: {accuracy:.2%}\n\nBy Dataset:\n" + "\n".join(f"  {ds}: {len(data['scores'])} examples, {data['correct']} correct" for ds, data in by_dataset.items()))),
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
                })
                
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
                
                # Use dataset-specific evaluator
                if self.dataset_type == "bizfinbench":
                    eval_result = self.dataset_evaluator.evaluate(
                        predicted=response,
                        expected=example.answer,
                        task_type=self.task_type,
                    )
                    result = {
                        "example_id": example.example_id,
                        "task_type": self.task_type,
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "expected": example.answer[:100] + "..." if len(example.answer) > 100 else example.answer,
                        "predicted": response[:200] + "..." if len(response) > 200 else response,
                        "score": eval_result.score,
                        "is_correct": eval_result.is_correct,
                        "feedback": eval_result.feedback,
                    }
                    
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
                    )
                    result = {
                        "example_id": example.example_id,
                        "category": example.category.value if hasattr(example.category, 'value') else str(example.category),
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "expected": example.answer[:100] + "..." if len(example.answer) > 100 else example.answer,
                        "predicted": response[:200] + "..." if len(response) > 200 else response,
                        "score": eval_result.score,
                        "is_correct": eval_result.score >= 0.8,  # Threshold for correctness
                        "correct_count": eval_result.correct_count,
                        "total_count": eval_result.total_count,
                        "feedback": eval_result.feedback,
                    }
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
        examples_to_eval = self._loaded_examples[:num_tasks] if num_tasks else self._loaded_examples
        
        for i, example in enumerate(examples_to_eval):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[{i+1}/{len(examples_to_eval)}] Evaluating {example.dataset_type}: {example.example_id}"
                )
            )
            
            try:
                # Send question to Purple Agent
                response = await self.messenger.talk_to_agent(
                    message=example.question,
                    url=purple_agent_url,
                    new_conversation=True,
                    timeout=self.eval_config.timeout_seconds if self.eval_config else 300,
                )
                
                # Get appropriate evaluator
                evaluator = self._evaluators.get(example.dataset_type)
                if not evaluator:
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
                    )
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "task_type": example.task_type,
                        "language": example.language,
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "expected": example.answer[:100] + "..." if len(example.answer) > 100 else example.answer,
                        "predicted": response[:200] + "..." if len(response) > 200 else response,
                        "score": eval_result.score,
                        "is_correct": eval_result.is_correct,
                        "feedback": eval_result.feedback,
                    }
                    
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
                    )
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "category": example.category,
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "expected": example.answer[:100] + "..." if len(example.answer) > 100 else example.answer,
                        "predicted": response[:200] + "..." if len(response) > 200 else response,
                        "score": eval_result.score,
                        "is_correct": eval_result.score >= 0.8,
                        "correct_count": eval_result.correct_count,
                        "total_count": eval_result.total_count,
                        "feedback": eval_result.feedback,
                    }
                    
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
                        "predicted_full": response[:200] + "..." if len(response) > 200 else response,
                        "score": 1.0 if is_correct else 0.0,
                        "is_correct": is_correct,
                        "feedback": f"Extracted: {extracted}, Expected: {expected}",
                    }
                    eval_result = type('obj', (object,), {'score': result['score']})()

                elif example.dataset_type == "options":
                    # Options Alpha Challenge evaluation
                    from cio_agent.models import AgentResponse

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

                    # Create a FABTask for the evaluator
                    fab_task = FABTask(
                        task_id=example.example_id,
                        category=task_category,
                        difficulty=TaskDifficulty.MEDIUM,
                        question=example.question,
                        ground_truth=GroundTruth(
                            macro_thesis=example.answer,
                            key_themes=[example.category],
                        ),
                        rubric=TaskRubric(criteria=[], penalty_conditions=[]),
                    )

                    # Create AgentResponse
                    agent_response = AgentResponse(
                        analysis=response,
                        recommendation=self._extract_recommendation(response),
                    )

                    # Initialize OptionsEvaluator with the task
                    options_evaluator = OptionsEvaluator(task=fab_task)
                    options_score = options_evaluator.evaluate(agent_response)

                    # Normalize score from 0-100 to 0-1
                    normalized_score = options_score.score / 100.0

                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "category": example.category,
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "expected": example.answer[:100] + "..." if len(example.answer) > 100 else example.answer,
                        "predicted": response[:200] + "..." if len(response) > 200 else response,
                        "score": normalized_score,
                        "score_raw": options_score.score,
                        "is_correct": options_score.score >= 70,  # 70/100 threshold
                        "pnl_accuracy": options_score.pnl_accuracy,
                        "greeks_accuracy": options_score.greeks_accuracy,
                        "strategy_quality": options_score.strategy_quality,
                        "risk_management": options_score.risk_management,
                        "feedback": options_score.feedback,
                    }
                    eval_result = type('obj', (object,), {'score': normalized_score})()

                else:
                    # Generic handling for unknown types
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "question": example.question[:200] + "..." if len(example.question) > 200 else example.question,
                        "predicted": response[:200] + "..." if len(response) > 200 else response,
                        "score": 0.0,  # No evaluator, no score
                        "is_correct": False,
                        "feedback": "No evaluator configured for this dataset type",
                    }
                
                # Optional debate
                if conduct_debate and eval_result.score > 0:
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


