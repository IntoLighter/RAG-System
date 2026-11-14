import time
import uuid
from collections.abc import Sequence
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from sqlmodel import col, delete, func, select, update
from sqlmodel.ext.asyncio.session import AsyncSession

from dependency.agent import get_agent, get_checkpointer
from dependency.db import engine, get_session
from model.chat_session import ChatSession
from schema.session import (
    MessageRequest,
    MessageResponse,
    RetrievedContext,
    SessionHistory,
    SessionInfo,
    SessionMessage,
    SessionUpdate,
)
from utils import elapsed_ms

logger = structlog.get_logger()

router = APIRouter(prefix="/sessions", tags=["sessions"])

TITLE_MAX_CHARS = 80

SESSION_NOT_FOUND = "Session not found"
NO_COMPLETED_TURNS = "Session has no answered messages"


def _config(session_id: UUID) -> RunnableConfig:
    return {"configurable": {"thread_id": str(session_id)}}


def _title(message: str) -> str:
    return " ".join(message.split())[:TITLE_MAX_CHARS]


async def _ensure_exists(session_id: UUID) -> None:
    """Check in a short transaction, so no connection is held during generation."""
    async with AsyncSession(engine) as db:
        result = await db.exec(
            select(ChatSession.id).where(ChatSession.id == session_id),
        )
        if result.first() is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, SESSION_NOT_FOUND)


async def _touch(session_id: UUID) -> None:
    async with AsyncSession(engine) as db:
        await db.exec(
            update(ChatSession)
            .where(col(ChatSession.id) == session_id)
            .values(updated_at=func.now()),
        )
        await db.commit()


async def _get_messages(
    agent: CompiledStateGraph,
    session_id: UUID,
) -> list[BaseMessage]:
    state = await agent.aget_state(_config(session_id))
    return state.values.get("messages", [])


async def _set_messages(
    agent: CompiledStateGraph,
    checkpointer: AsyncPostgresSaver,
    session_id: UUID,
    messages: Sequence[BaseMessage],
) -> None:
    if not messages:
        await checkpointer.adelete_thread(str(session_id))
        return
    await agent.aupdate_state(
        _config(session_id),
        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]},
    )


def _is_final_answer(message: BaseMessage) -> bool:
    return isinstance(message, AIMessage) and not message.tool_calls


def _completed_turns(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop the tail of a turn that failed before the final answer."""
    last_answer = max(
        (i for i, m in enumerate(messages) if _is_final_answer(m)),
        default=-1,
    )
    return messages[: last_answer + 1]


def _last_turn_start(messages: list[BaseMessage]) -> int:
    start = max(
        (
            i
            for i, m in enumerate(_completed_turns(messages))
            if isinstance(m, HumanMessage)
        ),
        default=None,
    )
    if start is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, NO_COMPLETED_TURNS)
    return start


def _retrieved(messages: list[BaseMessage]) -> list[RetrievedContext]:
    # Artifacts come back from the checkpoint as dicts, not RetrieveArtifact
    return [
        RetrievedContext.model_validate(m.artifact)
        for m in messages
        if isinstance(m, ToolMessage) and m.artifact is not None
    ]


async def _run_turn(
    agent: CompiledStateGraph,
    checkpointer: AsyncPostgresSaver,
    session_id: UUID,
    message: str,
) -> MessageResponse:
    """Answer a message; on failure leave the history as it was before it."""
    structlog.contextvars.bind_contextvars(session_id=str(session_id))
    human_message = HumanMessage(content=message, id=str(uuid.uuid4()))
    history = await _get_messages(agent, session_id)

    started_at = time.perf_counter()
    try:
        response = await agent.ainvoke(
            {"messages": [human_message]},
            config=_config(session_id),
            durability="exit",
        )
    except Exception:
        logger.warning("turn_failed_history_restored")
        await _set_messages(agent, checkpointer, session_id, history)
        raise
    agent_elapsed_ms = elapsed_ms(started_at)

    messages = response["messages"]
    new_messages = messages[len(history) + 1 :]
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

    return MessageResponse(
        session_id=session_id,
        answer=messages[-1].text,
        retrieved=_retrieved(new_messages),
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_session(
    body: MessageRequest,
    agent: Annotated[CompiledStateGraph, Depends(get_agent)],
    checkpointer: Annotated[AsyncPostgresSaver, Depends(get_checkpointer)],
) -> MessageResponse:
    """Start a conversation with its first message."""

    session_id = uuid.uuid4()
    result = await _run_turn(agent, checkpointer, session_id, body.message)
    async with AsyncSession(engine) as db:
        db.add(ChatSession(id=session_id, title=_title(body.message)))
        await db.commit()
    logger.info("session_created")
    return result


@router.get("")
async def list_sessions(
    db: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[SessionInfo]:
    """Most recently active first."""

    result = await db.exec(
        select(ChatSession)
        .order_by(col(ChatSession.updated_at).desc())
        .limit(limit)
        .offset(offset),
    )
    return [SessionInfo.model_validate(s) for s in result]


@router.post("/{session_id}/messages")
async def send_message(
    session_id: UUID,
    body: MessageRequest,
    agent: Annotated[CompiledStateGraph, Depends(get_agent)],
    checkpointer: Annotated[AsyncPostgresSaver, Depends(get_checkpointer)],
) -> MessageResponse:
    await _ensure_exists(session_id)
    result = await _run_turn(agent, checkpointer, session_id, body.message)
    await _touch(session_id)
    return result


@router.delete(
    "/{session_id}/messages/last",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_last_turn(
    session_id: UUID,
    agent: Annotated[CompiledStateGraph, Depends(get_agent)],
    checkpointer: Annotated[AsyncPostgresSaver, Depends(get_checkpointer)],
) -> Response:
    """Remove the last question and its answer."""

    await _ensure_exists(session_id)
    messages = await _get_messages(agent, session_id)
    start = _last_turn_start(messages)
    await _set_messages(agent, checkpointer, session_id, messages[:start])
    await _touch(session_id)
    logger.info("last_turn_deleted", session_id=str(session_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{session_id}/regenerate")
async def regenerate_answer(
    session_id: UUID,
    agent: Annotated[CompiledStateGraph, Depends(get_agent)],
    checkpointer: Annotated[AsyncPostgresSaver, Depends(get_checkpointer)],
) -> MessageResponse:
    """Ask the last question again, replacing its answer."""

    await _ensure_exists(session_id)
    messages = await _get_messages(agent, session_id)
    start = _last_turn_start(messages)
    await _set_messages(agent, checkpointer, session_id, messages[:start])
    try:
        result = await _run_turn(agent, checkpointer, session_id, messages[start].text)
    except Exception:
        await _set_messages(agent, checkpointer, session_id, messages)
        raise
    await _touch(session_id)
    return result


@router.get("/{session_id}")
async def get_session_history(
    session_id: UUID,
    agent: Annotated[CompiledStateGraph, Depends(get_agent)],
    *,
    include_sources: bool = False,
) -> SessionHistory:
    """Questions and final answers of every completed turn, without tool calls."""

    await _ensure_exists(session_id)
    history: list[SessionMessage] = []
    turn: list[BaseMessage] = []
    for m in _completed_turns(await _get_messages(agent, session_id)):
        if isinstance(m, HumanMessage):
            history.append(SessionMessage(role="user", content=m.text))
            turn = []
        elif _is_final_answer(m):
            history.append(
                SessionMessage(
                    role="assistant",
                    content=m.text,
                    retrieved=_retrieved(turn) if include_sources else None,
                ),
            )
        else:
            turn.append(m)
    return SessionHistory(session_id=session_id, messages=history)


@router.patch("/{session_id}")
async def rename_session(
    session_id: UUID,
    body: SessionUpdate,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> SessionInfo:
    chat_session = await db.get(ChatSession, session_id)
    if chat_session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, SESSION_NOT_FOUND)
    chat_session.title = body.title
    await db.commit()
    await db.refresh(chat_session)
    return SessionInfo.model_validate(chat_session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    checkpointer: Annotated[AsyncPostgresSaver, Depends(get_checkpointer)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    result = await db.exec(delete(ChatSession).where(col(ChatSession.id) == session_id))
    if result.rowcount == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, SESSION_NOT_FOUND)
    await checkpointer.adelete_thread(str(session_id))
    await db.commit()
    logger.info("session_deleted", session_id=str(session_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
