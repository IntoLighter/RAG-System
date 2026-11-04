from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Response, status
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph

from dependency.agent import get_agent, get_checkpointer
from schema.session import SessionHistory, SessionMessage

logger = structlog.get_logger()

router = APIRouter(prefix="/sessions", tags=["sessions"])


async def _get_messages(
    agent: CompiledStateGraph,
    session_id: UUID,
) -> list[BaseMessage]:
    state = await agent.aget_state({"configurable": {"thread_id": str(session_id)}})
    messages = state.values.get("messages")
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found")
    return messages


@router.get("/{session_id}")
async def get_session_history(
    session_id: UUID,
    agent: Annotated[CompiledStateGraph, Depends(get_agent)],
) -> SessionHistory:
    """Questions and final answers of every turn, without tool calls."""

    messages = await _get_messages(agent, session_id)
    return SessionHistory(
        session_id=session_id,
        messages=[
            SessionMessage(
                role="user" if isinstance(m, HumanMessage) else "assistant",
                content=m.text,
            )
            for m in messages
            if isinstance(m, HumanMessage)
            or (isinstance(m, AIMessage) and not m.tool_calls)
        ],
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    agent: Annotated[CompiledStateGraph, Depends(get_agent)],
    checkpointer: Annotated[AsyncPostgresSaver, Depends(get_checkpointer)],
) -> Response:
    await _get_messages(agent, session_id)
    await checkpointer.adelete_thread(str(session_id))
    logger.info("session_deleted", session_id=str(session_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
