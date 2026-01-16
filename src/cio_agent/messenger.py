"""
A2A Messaging Utilities for Green Agent

Provides utilities for communicating with Purple Agents via the A2A protocol.
Based on the official green-agent-template from RDI-Foundation.
"""

import json
import logging
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)
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
    Caches A2A clients to avoid repeated agent card fetches.
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self._context_ids = {}
        self._clients = {}  # Cache A2A clients per URL
        self._timeout = timeout
        self._httpx_client = None

    async def _get_httpx_client(self) -> httpx.AsyncClient:
        """Get or create the shared httpx client."""
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=self._timeout)
        return self._httpx_client

    async def _get_a2a_client(self, url: str):
        """Get A2A client for URL, creating and caching if needed."""
        if url not in self._clients:
            httpx_client = await self._get_httpx_client()
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
            agent_card = await resolver.get_agent_card()
            config = ClientConfig(httpx_client=httpx_client, streaming=False)
            factory = ClientFactory(config)
            self._clients[url] = factory.create(agent_card)
        return self._clients[url]

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
        # Log the outgoing question
        msg_preview = message[:200] + "..." if len(message) > 200 else message
        logger.info(f"[QUESTION] Sending to {url}: {msg_preview}")

        client = await self._get_a2a_client(url)

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

        # Log the response
        resp_preview = outputs["response"][:200] + "..." if len(outputs["response"]) > 200 else outputs["response"]
        logger.info(f"[RESPONSE] From {url}: {resp_preview}")

        return outputs["response"]

    async def close(self):
        """Close the httpx client."""
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None

    def reset(self):
        """Reset all conversation contexts and client cache."""
        self._context_ids = {}
        self._clients = {}
