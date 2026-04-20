"""
Stage 2 (workaround): Add Foundry IQ — ground answers in an enterprise
knowledge base served by Azure AI Search via its MCP endpoint.

This version uses a manual HTTP wrapper around the MCP endpoint because the
KB MCP server currently returns resource content with `uri: null`, which
the standard MCP SDK rejects. Once that's fixed upstream, see
stage2_foundry_iq.py for the clean MCPStreamableHTTPTool version.

What changes from Stage 1:
    - Add a `KnowledgeBaseMCPTool` that calls the KB MCP endpoint over HTTP.
    - Pass `kb_tool.retrieve` as an additional tool on the Agent.
    - Update the system prompt to prefer the KB.

Prerequisites (in addition to Stage 1):
    AZURE_AI_SEARCH_SERVICE_ENDPOINT=https://<your-search>.search.windows.net
    AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME=zava-company-kb

Run:
    python stage2_foundry_iq_workaround.py
"""

import asyncio
import json
import logging
import os
from datetime import date
from typing import Annotated

import httpx
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from pydantic import Field

load_dotenv(override=True)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("stage2")
logger.setLevel(logging.INFO)


@tool
def get_enrollment_deadline_info() -> dict:
    """Return enrollment timeline details for health insurance plans."""
    logger.info("[tool] get_enrollment_deadline_info()")
    return {
        "benefits_enrollment_opens": "2026-11-11",
        "benefits_enrollment_closes": "2026-11-30",
    }


class KnowledgeBaseMCPTool:
    """Wraps the KB MCP endpoint as a callable tool, working around the MCP
    SDK validation bug where the endpoint returns resource content with
    uri: null.
    """

    def __init__(self, http_client: httpx.Client, mcp_url: str) -> None:
        self._http_client = http_client
        self._mcp_url = mcp_url
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Perform MCP handshake if not already done."""
        if self._initialized:
            return
        self._http_client.post(
            self._mcp_url,
            json={
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"sampling": {}},
                    "clientInfo": {"name": "stage2-agent", "version": "0.1.0"},
                },
            },
            headers=self._headers,
        )
        self._http_client.post(
            self._mcp_url,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=self._headers,
        )
        self._initialized = True

    def retrieve(
        self,
        queries: Annotated[
            list[str],
            Field(
                description=(
                    "1 to 4 concise search queries (max ~12 words each). "
                    "Include facets and alternative formulations as separate entries."
                ),
                min_length=1,
                max_length=4,
            ),
        ],
    ) -> str:
        """Search the Zava company knowledge base for HR policies, benefits,
        insurance, and job information.

        Use this tool to find information from internal company documents
        before answering HR-related questions.
        """
        self._ensure_initialized()
        logger.info("[tool] kb.retrieve(queries=%s)", queries)
        response = self._http_client.post(
            self._mcp_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "knowledge_base_retrieve",
                    "arguments": {"queries": queries},
                },
            },
            headers=self._headers,
        )
        for line in response.text.split("\n"):
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:].strip())
            if "result" not in data or "content" not in data["result"]:
                continue
            snippets = []
            for item in data["result"]["content"]:
                if item.get("type") == "resource" and "resource" in item:
                    snippets.append(item["resource"].get("text", ""))
                elif item.get("type") == "text":
                    snippets.append(item.get("text", ""))
            return "\n\n---\n\n".join(snippets)
        return "No results found."


async def main():
    credential = DefaultAzureCredential()

    # --- Chat client (Foundry / Azure OpenAI) -----------------------------
    aoai_token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    client = OpenAIChatClient(
        base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT']}/openai/v1/",
        api_key=aoai_token_provider,
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
    )

    # --- Foundry IQ knowledge base via MCP --------------------------------
    mcp_url = (
        f"{os.environ['AZURE_AI_SEARCH_SERVICE_ENDPOINT']}"
        f"/knowledgebases/{os.environ['AZURE_AI_SEARCH_KNOWLEDGE_BASE_NAME']}"
        f"/mcp?api-version=2025-11-01-Preview"
    )

    def _add_search_auth(request: httpx.Request) -> None:
        token = credential.get_token("https://search.azure.com/.default")
        request.headers["Authorization"] = f"Bearer {token.token}"

    http_client = httpx.Client(
        event_hooks={"request": [_add_search_auth]},
        timeout=httpx.Timeout(30.0, read=300.0),
    )
    kb_tool = KnowledgeBaseMCPTool(http_client, mcp_url)

    agent = Agent(
        client=client,
        instructions=(
            f"You are an internal HR helper for Zava. Today's date is {date.today().isoformat()}. "
            "Use the knowledge base tool to answer questions about HR policies, benefits, "
            "and company information, and ground all answers in the retrieved context. "
            "Use get_enrollment_deadline_info for benefits enrollment timing. "
            "If you cannot answer from the tools, say so clearly."
        ),
        tools=[kb_tool.retrieve, get_enrollment_deadline_info],
    )

    response = await agent.run(
        "What PerksPlus benefits are there, and when do I need to enroll by?"
    )
    print("\n--- Agent answer ---")
    print(response.text)

    http_client.close()


if __name__ == "__main__":
    asyncio.run(main())
