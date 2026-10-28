import uuid

from fastapi import APIRouter
from langchain_core.messages import HumanMessage
from langchain_postgres import PostgresChatMessageHistory

from dependency.agent import agent
from dependency.db import connection_pool
from dependency.settings import settings
from schema.query import QueryRequest, QueryResponse, RetrievedContext

router = APIRouter(tags=["query"])


@router.post(path="/query")
async def query(query: QueryRequest) -> QueryResponse:
    session_id = str(query.session_id or uuid.uuid4())
    human_message = HumanMessage(content=query.message)

    async with connection_pool.connection() as conn:
        history = await PostgresChatMessageHistory(
            settings.postgres.chat_table_name,
            session_id,
            async_connection=conn,
        ).aget_messages()

    response = await agent.ainvoke({"messages": [*history, human_message]})
    answer = response["messages"][-1]

    async with connection_pool.connection() as conn:
        await PostgresChatMessageHistory(
            settings.postgres.chat_table_name,
            session_id,
            async_connection=conn,
        ).aadd_messages([human_message, answer])

    new_messages = response["messages"][len(history) + 1 :]
    artifacts = [message.artifact for message in new_messages if message.type == "tool"]

    return QueryResponse(
        session_id=session_id,
        answer=answer.text,
        retrieved=[
            RetrievedContext(query=artifact.query, chunks=artifact.chunks)
            for artifact in artifacts
        ],
    )
