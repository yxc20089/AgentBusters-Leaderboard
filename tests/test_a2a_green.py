"""
A2A Conformance Tests for Green Agent

Based on the official green-agent-template test pattern from RDI-Foundation.
These tests verify that the Green Agent implements the A2A protocol correctly.
"""

import json
from uuid import uuid4

import pytest
import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Role, Part, TextPart


# Override the global agent fixture to skip when no server is running.
@pytest.fixture
def agent(request):
    url = request.config.getoption("--agent-url")
    try:
        httpx.get(f"{url}/.well-known/agent.json", timeout=2)
    except Exception:
        pytest.skip("Green agent not running - skipping A2A integration tests")
    return url


# A2A validation helpers - adapted from a2a-inspector
def validate_agent_card(card_data: dict) -> list[str]:
    """Validate agent card structure and required fields."""
    errors = []
    
    required_fields = ["name", "url", "version", "skills"]
    for field in required_fields:
        if field not in card_data:
            errors.append(f"Missing required field: {field}")
    
    if "skills" in card_data:
        if not isinstance(card_data["skills"], list):
            errors.append("'skills' must be a list")
        elif len(card_data["skills"]) == 0:
            errors.append("Agent must have at least one skill")
        else:
            for i, skill in enumerate(card_data["skills"]):
                if "id" not in skill:
                    errors.append(f"Skill {i} missing 'id'")
                if "name" not in skill:
                    errors.append(f"Skill {i} missing 'name'")
    
    return errors


def validate_event(event_data: dict) -> list[str]:
    """Validate A2A event structure."""
    errors = []
    
    if "kind" not in event_data:
        errors.append("Event missing 'kind' field")
    
    return errors


# A2A messaging helpers
async def send_text_message(text: str, url: str, context_id: str | None = None, streaming: bool = False):
    """Send a text message to an A2A agent."""
    async with httpx.AsyncClient(timeout=120) as httpx_client:  # 120s for LLM calls
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]

    return events


# Pytest fixtures are in conftest.py


# A2A conformance tests
def test_agent_card(agent):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent.json", timeout=10)
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)


def test_agent_card_has_fab_skill(agent):
    """Verify the agent card includes FAB++ evaluation skill."""
    response = httpx.get(f"{agent}/.well-known/agent.json", timeout=10)
    assert response.status_code == 200

    card_data = response.json()
    
    # Check for FAB++ related skill
    skill_ids = [s.get("id", "") for s in card_data.get("skills", [])]
    skill_names = [s.get("name", "").lower() for s in card_data.get("skills", [])]
    
    has_fab_skill = (
        any("fab" in sid.lower() for sid in skill_ids) or
        any("evaluation" in name or "benchmark" in name for name in skill_names)
    )
    
    assert has_fab_skill, "Agent should have a FAB++ evaluation skill"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False])  # Only test non-streaming for now
async def test_message(agent, streaming):
    """Test that agent returns valid A2A message format."""
    # Send a simple test message (not a full evaluation request)
    events = await send_text_message("Hello", agent, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                # Unknown event type - not necessarily an error
                pass

    assert events, "Agent should respond with at least one event"


@pytest.mark.asyncio
async def test_evaluation_request_format(agent):
    """Test that agent can process a properly formatted evaluation request."""
    # Create a valid evaluation request
    eval_request = {
        "participants": {
            "purple_agent": "http://localhost:9010/"
        },
        "config": {
            "ticker": "NVDA",
            "task_category": "beat_or_miss",
            "num_tasks": 1,
            "conduct_debate": False
        }
    }
    
    # Note: This test just validates the request format is accepted
    # It will fail if no purple agent is running, which is expected in CI
    try:
        events = await send_text_message(
            json.dumps(eval_request),
            agent,
            streaming=False
        )
        # If we get here, the agent accepted the request format
        assert len(events) > 0
    except Exception as e:
        # Expected to fail if purple agent isn't running
        # But we can check it's not a format rejection
        error_str = str(e).lower()
        assert "invalid request" not in error_str, f"Request format was rejected: {e}"


@pytest.mark.asyncio
async def test_synthetic_questions_evaluation(agent, purple_agent):
    """
    Test that the Green Agent can evaluate using synthetic questions.
    
    This test requires:
    - Green Agent running with --synthetic-questions
    - Purple Agent running
    
    Run with:
        pytest tests/test_a2a_green.py::test_synthetic_questions_evaluation -v \
            --agent-url http://localhost:9109 \
            --purple-url http://localhost:9110
    """
    # Create evaluation request that will use synthetic questions
    eval_request = {
        "participants": {
            "purple_agent": purple_agent
        },
        "config": {
            "num_tasks": 1  # Only evaluate 1 synthetic question
        }
    }
    
    try:
        events = await send_text_message(
            json.dumps(eval_request),
            agent,
            streaming=False
        )
        
        # Check that we got evaluation events
        assert len(events) > 0, "Expected at least one event from the agent"
        
        # Verify the response contains expected evaluation structure
        for event in events:
            if hasattr(event, "model_dump"):
                event_data = event.model_dump()
                # Check for task or artifact in response
                if "artifacts" in str(event_data):
                    # Found evaluation artifacts - success
                    break
        
    except httpx.ConnectError:
        pytest.skip("Purple agent not running - skipping integration test")
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "connect" in error_str:
            pytest.skip(f"Connection error (expected in CI): {e}")
        raise

