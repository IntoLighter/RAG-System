from langchain_openai import OpenAIEmbeddings
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.indexes import HNSWIndex
from langchain_postgres.v2.vectorstores import PGVectorStore
from psycopg.conninfo import conninfo_to_dict

from dependency.db import connection_pool
from dependency.settings import settings

VECTOR_STORE_NOT_INITIALIZED = "PostgreSQL vector store has not been initialized"

pg_engine = PGEngine.from_connection_string(
    "postgresql+psycopg://",
    connect_args=conninfo_to_dict(settings.postgres.conninfo),
    pool_pre_ping=True,
)

embedding = OpenAIEmbeddings(
    model=settings.embedding.name,
    base_url=settings.embedding.base_url,
    api_key="EMPTY",
    check_embedding_ctx_length=False,
)

vector_store: PGVectorStore | None = None


async def _table_exists(table_name: str, schema_name: str = "public") -> bool:
    async with connection_pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT to_regclass(format('%%I.%%I', %s::text, %s::text))",
            (schema_name, table_name),
        )
        row = await cursor.fetchone()
    return row is not None and row[0] is not None


async def initialize_vector_store() -> None:

    global vector_store  # noqa: PLW0603

    if not await _table_exists(settings.postgres.embedding_table_name):
        await pg_engine.ainit_vectorstore_table(
            table_name=settings.postgres.embedding_table_name,
            vector_size=settings.postgres.vector_size,
        )

    vector_store = await PGVectorStore.create(
        pg_engine,
        embedding_service=embedding,
        table_name=settings.postgres.embedding_table_name,
    )

    if not await vector_store.ais_valid_index():
        await vector_store.aapply_vector_index(HNSWIndex())
