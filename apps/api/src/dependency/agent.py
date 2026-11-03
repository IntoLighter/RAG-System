import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from openai import AsyncOpenAI
from pydantic import BaseModel, field_validator

from dependency.db import connection_pool
from dependency.settings import settings
from dependency.vector_store import get_vector_store
from schema.query import RetrievedChunk
from utils import elapsed_ms

logger = structlog.get_logger()

DOCUMENT_PREVIEW_CHARS = 100


client = AsyncOpenAI(
    base_url=settings.reranker.base_url,
    api_key="EMPTY",
)


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str

    @field_validator("document", mode="before")
    @classmethod
    def extract_document(cls, v: Any) -> str:
        return v.get("text")


class RerankResponse(BaseModel):
    results: list[RerankResult]


async def rerank(query: str, docs: list[str]) -> list[RerankResult]:
    raw = await client.post(
        "/rerank",
        cast_to=httpx.Response,
        body={
            "model": settings.reranker.name,
            "query": query,
            "documents": docs,
        },
    )

    response = RerankResponse.model_validate(raw.json())
    return response.results


@dataclass
class RetrieveArtifact:
    query: str
    chunks: list[RetrievedChunk]


def _format_chunk(number: int, chunk: RetrievedChunk) -> str:
    source = chunk.filename or "unknown source"
    if chunk.page is not None:
        source += f", p. {chunk.page}"
    return f"[Document {number}] ({source})\n{chunk.content}"


@tool(response_format="content_and_artifact")
async def retrieve_knowledge(
    query: str,
) -> tuple[str, RetrieveArtifact]:
    """
    Search the knowledge base of uploaded books for passages about a topic.

    Use it for questions about facts, terms, events or ideas from the books.
    Do not use it for questions about this conversation itself.
    `query` must be a standalone search query: resolve pronouns and references
    to earlier messages, e.g. "tell me more about it" -> "quest structure".
    """

    task_description = (
        "Given a user question, retrieve relevant passages "
        "from a knowledge base that answer the question."
    )
    final_query = f"Instruct: {task_description}\nQuery: {query}"

    started_at = time.perf_counter()
    chunks = await get_vector_store().asimilarity_search(
        final_query,
        k=settings.rag.retrieve_top_k,
    )
    search_elapsed_ms = elapsed_ms(started_at)

    logger.info(
        "semantic_search_finished",
        query=query,
        count=len(chunks),
        elapsed_ms=search_elapsed_ms,
    )

    doc_texts = [d.page_content for d in chunks]

    started_at = time.perf_counter()
    reranked_documents = await rerank(query, doc_texts)
    rerank_elapsed_ms = elapsed_ms(started_at)

    relevant_docs = [
        d
        for d in reranked_documents
        if d.relevance_score >= settings.rag.rerank_min_score
    ]
    rerank_top_k_docs = relevant_docs[: settings.rag.rerank_top_k]

    logger.info(
        "rerank_finished",
        count=len(reranked_documents),
        relevant=len(relevant_docs),
        returned=len(rerank_top_k_docs),
        top_score=max(
            (result.relevance_score for result in reranked_documents),
            default=None,
        ),
        elapsed_ms=rerank_elapsed_ms,
    )
    logger.debug(
        "rerank_documents",
        documents=[
            {
                "search_rank": result.index,
                "score": round(result.relevance_score, 3),
                "preview": result.document[:DOCUMENT_PREVIEW_CHARS],
            }
            for result in reranked_documents
        ],
    )

    retrieved = [
        RetrievedChunk(
            content=d.document,
            filename=chunks[d.index].metadata.get("filename"),
            page=chunks[d.index].metadata.get("page"),
            score=d.relevance_score,
        )
        for d in rerank_top_k_docs
    ]
    if not retrieved:
        response = "No relevant passages were found in the knowledge base."
    else:
        response = "\n\n".join(
            _format_chunk(i, chunk) for i, chunk in enumerate(retrieved, start=1)
        )

    return response, RetrieveArtifact(
        query=query,
        chunks=retrieved,
    )


prompt = """
You are an assistant that answers questions about uploaded books.
You have one tool, `retrieve_knowledge`, which searches the books.

Before answering, decide which kind of message the user sent.

A. A question about the content of the books: a fact, term, definition,
   character, event or idea. Examples:
   - "What is a quest?"
   - "Что такое квест?"
   - "How does the author describe the main conflict?"
   - "Tell me more about the second type" (a follow-up that needs new facts)
   For these you MUST call `retrieve_knowledge` before answering, even if
   you think you know the answer. Answer only from the retrieved passages.
   If they contain nothing relevant, say the information was not found
   in the knowledge base. Do not use your own knowledge.

B. A message about this conversation itself: what the user asked before,
   what you answered, a request to repeat, shorten, translate or rephrase
   a previous answer, greetings and thanks. Examples:
   - "What did I ask before?"
   - "Что я спрашивал раньше?"
   - "Summarize your last answer in one sentence"
   - "Thanks!"
   For these DO NOT call any tool. Answer directly from the conversation
   history above.

If you are unsure, treat the message as kind A.

When calling `retrieve_knowledge`, write a standalone search query: replace
pronouns and references like "it" or "the second one" with the actual topic
from the conversation.

Each retrieved passage is labeled with its source, e.g.
"[Document 1] (book.pdf, p. 42)". After each fact taken from a passage,
cite the source in parentheses, e.g. "(book.pdf, p. 42)".

Always answer in the language of the user's last message.
"""

generative = ChatOpenAI(
    model=settings.generative.name,
    base_url=settings.generative.base_url,
    api_key="EMPTY",
    temperature=settings.generative.temperature,
    top_p=settings.generative.top_p,
    max_tokens=settings.generative.max_tokens,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": settings.generative.enable_thinking,
        },
    },
)


AGENT_NOT_INITIALIZED = "Agent is not initialized, call init_agent"
CHECKPOINT_SCHEMA_OUTDATED = (
    "Checkpoint tables are at version {actual}, langgraph expects {expected}: "
    "add a migration for the new langgraph-checkpoint-postgres version"
)

_agent: CompiledStateGraph | None = None


async def _check_checkpoint_schema() -> None:
    """Fail fast if a langgraph upgrade brought migrations the database lacks."""
    expected = len(AsyncPostgresSaver.MIGRATIONS) - 1
    async with connection_pool.connection() as conn:
        cursor = await conn.execute("SELECT max(v) AS v FROM checkpoint_migrations")
        row = await cursor.fetchone()
    actual = row["v"] if row else None
    if actual != expected:
        raise RuntimeError(
            CHECKPOINT_SCHEMA_OUTDATED.format(actual=actual, expected=expected),
        )


async def init_agent() -> None:
    global _agent  # noqa: PLW0603
    await _check_checkpoint_schema()
    _agent = create_agent(
        generative,
        [retrieve_knowledge],
        system_prompt=prompt,
        checkpointer=AsyncPostgresSaver(connection_pool),
    )


def get_agent() -> CompiledStateGraph:
    if _agent is None:
        raise RuntimeError(AGENT_NOT_INITIALIZED)
    return _agent
