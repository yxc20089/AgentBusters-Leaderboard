"""
A2A Messaging Utilities for Green Agent

Provides utilities for communicating with Purple Agents via the A2A protocol.
Based on the official green-agent-template from RDI-Foundation.
"""

import json
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
    Consumer,
)
from a2a.types import (
    Message,
    Part,
    Role,
    TextPart,
    DataPart,
)


DEFAULT_TIMEOUT = 300


def create_message(
    *, role: Role = Role.user, text: str, context_id: str | None = None
) -> Message:
    """Create an A2A message."""
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def merge_parts(parts: list[Part]) -> str:
    """Merge message parts into a single string."""
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)


async def send_message(
    message: str,
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    consumer: Consumer | None = None,
):
    """
    Send a message to an A2A agent and return the response.
    
    Returns:
        dict with context_id, response and status (if exists)
    """
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=streaming,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        if consumer:
            await client.add_event_consumer(consumer)

        outbound_msg = create_message(text=message, context_id=context_id)
        last_event = None
        outputs = {"response": "", "context_id": None}

        # if streaming == False, only one event is generated
        async for event in client.send_message(outbound_msg):
            last_event = event

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] += merge_parts(msg.parts)

            case (task, update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                msg = task.status.message
                if msg:
                    outputs["response"] += merge_parts(msg.parts)
                if task.artifacts:
                    for artifact in task.artifacts:
                        outputs["response"] += merge_parts(artifact.parts)

            case _:
                pass

        return outputs


class Messenger:
    """
    A2A Messenger for communicating with Purple Agents.

    Maintains conversation context across multiple message exchanges.
    Caches agent cards to avoid repeated fetches.
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self._context_ids = {}
        self._agent_cards = {}
        self._timeout = timeout
        self._httpx_client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared httpx client."""
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=self._timeout)
        return self._httpx_client

    async def _get_agent_card(self, url: str):
        """Get agent card, using cache if available."""
        if url not in self._agent_cards:
            httpx_client = await self._get_client()
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
            self._agent_cards[url] = await resolver.get_agent_card()
        return self._agent_cards[url]

    async def talk_to_agent(
        self,
        message: str,
        url: str,
        new_conversation: bool = False,
        timeout: int | None = None,
    ) -> str:
        """
        Send a message to a Purple Agent and receive their response.

        Args:
            message: The message to send to the agent
            url: The agent's URL endpoint
            new_conversation: If True, start fresh conversation; if False, continue existing
            timeout: Timeout in seconds for the request (default: uses instance timeout)

        Returns:
            str: The agent's response message
        """
        httpx_client = await self._get_client()
        agent_card = await self._get_agent_card(url)

        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=False,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        context_id = None if new_conversation else self._context_ids.get(url, None)
        outbound_msg = create_message(text=message, context_id=context_id)

        last_event = None
        outputs = {"response": "", "context_id": None}

        async for event in client.send_message(outbound_msg):
            last_event = event

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] += merge_parts(msg.parts)

            case (task, update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                msg = task.status.message
                if msg:
                    outputs["response"] += merge_parts(msg.parts)
                if task.artifacts:
                    for artifact in task.artifacts:
                        outputs["response"] += merge_parts(artifact.parts)

            case _:
                pass

        if outputs.get("status", "completed") != "completed":
            raise RuntimeError(f"{url} responded with: {outputs}")
        self._context_ids[url] = outputs.get("context_id", None)
        return outputs["response"]

    async def close(self):
        """Close the httpx client."""
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None

    def reset(self):
        """Reset all conversation contexts and agent card cache."""
        self._context_ids = {}
        self._agent_cards = {}
