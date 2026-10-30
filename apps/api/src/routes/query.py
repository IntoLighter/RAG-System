import time
import uuid

import structlog
from fastapi import APIRouter
from langchain_core.messages import AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory

from dependency.agent import agent
from dependency.db import connection_pool
from dependency.settings import settings
from schema.query import QueryRequest, QueryResponse, RetrievedContext
from utils import elapsed_ms

logger = structlog.get_logger()

router = APIRouter(tags=["query"])


@router.post(path="/query")
async def query(query: QueryRequest) -> QueryResponse:
    session_id = str(query.session_id or uuid.uuid4())
    structlog.contextvars.bind_contextvars(session_id=session_id)
    human_message = HumanMessage(content=query.message)

    async with connection_pool.connection() as conn:
        history = await PostgresChatMessageHistory(
            settings.postgres.chat_table_name,
            session_id,
            async_connection=conn,
        ).aget_messages()

    started_at = time.perf_counter()
    response = await agent.ainvoke({"messages": [*history, human_message]})
    agent_elapsed_ms = elapsed_ms(started_at)
    answer = response["messages"][-1]

    new_messages = response["messages"][len(history) + 1 :]
    llm_messages = [m for m in new_messages if isinstance(m, AIMessage)]
    logger.info(
        "agent_finished",
        history_messages=len(history),
        llm_calls=len(llm_messages),
        tool_calls=sum(len(m.tool_calls) for m in llm_messages),
        input_tokens=sum(
            m.usage_metadata["input_tokens"] for m in llm_messages if m.usage_metadata
        ),
        output_tokens=sum(
            m.usage_metadata["output_tokens"] for m in llm_messages if m.usage_metadata
        ),
        elapsed_ms=agent_elapsed_ms,
    )

    async with connection_pool.connection() as conn:
        await PostgresChatMessageHistory(
            settings.postgres.chat_table_name,
            session_id,
            async_connection=conn,
        ).aadd_messages([human_message, answer])

    artifacts = [message.artifact for message in new_messages if message.type == "tool"]

    return QueryResponse(
        session_id=session_id,
        answer=answer.text,
        retrieved=[
            RetrievedContext(query=artifact.query, chunks=artifact.chunks)
            for artifact in artifacts
        ],
    )
