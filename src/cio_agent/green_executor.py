"""
A2A Executor for Green Agent

Handles incoming A2A requests and routes them to the Green Agent.
Based on the official green-agent-template from RDI-Foundation.
"""

from typing import Any, Optional

from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    UnsupportedOperationError,
    InvalidParamsError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
)

from cio_agent.green_agent import GreenAgent


class EvalRequest(BaseModel):
    """Request from agentbeats-client with participants and config."""
    participants: dict[str, HttpUrl]
    config: dict[str, Any] = {}


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


class GreenAgentExecutor(AgentExecutor):
    """
    A2A AgentExecutor for the FAB++ Green Agent.

    Manages agent instances per context and handles the A2A protocol
    lifecycle for assessment requests.
    """

    def __init__(
        self,
        eval_config: Optional[str] = None,
        synthetic_questions: Optional[list[dict]] = None,
        dataset_type: str = "synthetic",
        dataset_path: Optional[str] = None,
        task_type: Optional[str] = None,
        language: str = "en",
        limit: Optional[int] = None,
    ):
        """
        Initialize the executor.

        Args:
            eval_config: Path to evaluation config YAML file (recommended).
                        When provided, other dataset params are ignored.
            synthetic_questions: Optional list of synthetic questions to use
                                for evaluation instead of generating new ones.
            dataset_type: Type of dataset ('synthetic', 'bizfinbench', 'public_csv')
            dataset_path: Path to dataset directory or file
            task_type: For BizFinBench, the specific task type to evaluate
            language: Language for BizFinBench ('en' or 'cn')
            limit: Optional limit on number of examples
        """
        self.agents: dict[str, GreenAgent] = {}  # context_id to agent instance
        self.eval_config = eval_config
        self.synthetic_questions = synthetic_questions
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.language = language
        self.limit = limit

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute an assessment request.

        Args:
            context: The A2A request context
            event_queue: Queue for publishing task events
        """
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        # Parse EvalRequest from agentbeats-client
        request_text = context.get_user_input()
        try:
            eval_request = EvalRequest.model_validate_json(request_text)
        except ValidationError as e:
            raise ServerError(error=InvalidParamsError(message=f"Invalid request: {e}"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already processed (state: {task.status.state})"
                )
            )

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.get(context_id)
        if not agent:
            agent = GreenAgent(
                eval_config=self.eval_config,
                synthetic_questions=self.synthetic_questions,
                dataset_type=self.dataset_type,
                dataset_path=self.dataset_path,
                task_type=self.task_type,
                language=self.language,
                limit=self.limit,
            )
            self.agents[context_id] = agent

        updater = TaskUpdater(event_queue, task.id, context_id)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting assessment.\n{eval_request.model_dump_json()}",
                context_id=context_id,
            ),
        )

        try:
            await agent.run_eval(eval_request, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as e:
            print(f"Task failed with agent error: {e}")
            await updater.failed(
                new_agent_text_message(
                    f"Agent error: {e}",
                    context_id=context_id,
                    task_id=task.id
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel an ongoing task (not supported)."""
        raise ServerError(error=UnsupportedOperationError())
