import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic import BaseModel, field_validator

from dependency.settings import settings
from dependency.vector_store import vector_store
from utils import elapsed_ms

logger = structlog.get_logger()


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
    chunks: list[str]


async def get_hyde_query(query: str) -> str:
    hyde_prompt = f"""
    Write a detailed textbook-style passage that answers the question.

    Question: {query}

    Answer:
    """
    return await generative.ainvoke(hyde_prompt).content


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
    chunks = await vector_store.asimilarity_search(
        final_query,
        k=settings.rag.retrieve_top_k,
    )
    search_elapsed_ms = elapsed_ms(started_at)

    logger.info(
        "semantic_search_finished",
        query=final_query,
        count=len(chunks),
        elapsed_ms=search_elapsed_ms,
    )
    logger.debug(
        "semantic_search_documents",
        query=final_query,
        documents=[doc.page_content for doc in chunks],
    )

    doc_texts = [d.page_content for d in chunks]

    started_at = time.perf_counter()
    reranked_documents = await rerank(query, doc_texts)
    rerank_elapsed_ms = elapsed_ms(started_at)

    logger.info(
        "rerank_finished",
        query=query,
        count=len(reranked_documents),
        top_score=max(
            (result.relevance_score for result in reranked_documents),
            default=None,
        ),
        elapsed_ms=rerank_elapsed_ms,
    )
    logger.debug(
        "rerank_documents",
        query=query,
        documents=[
            {
                "index": result.index,
                "score": result.relevance_score,
                "content": result.document,
            }
            for result in reranked_documents
        ],
    )

    rerank_top_k_docs = reranked_documents[: settings.rag.rerank_top_k]
    rerank_top_k_texts = [d.document for d in rerank_top_k_docs]

    chunks = rerank_top_k_texts
    response = "\n\n".join(
        f"[Document {i}]\n{doc}" for i, doc in enumerate(chunks, start=1)
    )
    return response, RetrieveArtifact(
        query=query,
        chunks=chunks,
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

Always answer in the language of the user's last message.
"""

generative = ChatOpenAI(
    model=settings.generative.name,
    base_url=settings.generative.base_url,
    api_key="EMPTY",
)

agent = create_agent(generative, [retrieve_knowledge], system_prompt=prompt)
