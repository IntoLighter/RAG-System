import time
import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from dependency.agent import RetrieveArtifact, get_agent
from schema.query import QueryRequest, QueryResponse, RetrievedContext
from utils import elapsed_ms

logger = structlog.get_logger()

router = APIRouter(tags=["query"])


@router.post(path="/query")
async def query(
    query: QueryRequest,
    agent: Annotated[CompiledStateGraph, Depends(get_agent)],
) -> QueryResponse:
    session_id = str(query.session_id or uuid.uuid4())
    structlog.contextvars.bind_contextvars(session_id=session_id)
    human_message = HumanMessage(content=query.message, id=str(uuid.uuid4()))

    started_at = time.perf_counter()
    response = await agent.ainvoke(
        {"messages": [human_message]},
        config={"configurable": {"thread_id": session_id}},
    )
    agent_elapsed_ms = elapsed_ms(started_at)

    messages = response["messages"]
    history_len = next(i for i, m in enumerate(messages) if m.id == human_message.id)
    new_messages = messages[history_len + 1 :]
    answer = messages[-1]

    llm_messages = [m for m in new_messages if isinstance(m, AIMessage)]
    logger.info(
        "agent_finished",
        history_messages=history_len,
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

    artifacts = [
        m.artifact
        for m in new_messages
        if isinstance(m, ToolMessage) and isinstance(m.artifact, RetrieveArtifact)
    ]

    return QueryResponse(
        session_id=session_id,
        answer=answer.text,
        retrieved=[
            RetrievedContext(query=artifact.query, chunks=artifact.chunks)
            for artifact in artifacts
        ],
    )
