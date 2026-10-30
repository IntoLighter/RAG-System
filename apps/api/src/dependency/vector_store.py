from langchain_openai import OpenAIEmbeddings
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.vectorstores import PGVectorStore
from psycopg.conninfo import conninfo_to_dict

from dependency.settings import settings

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

VECTOR_STORE_NOT_INITIALIZED = "Vector store is not initialized, call init_vector_store"

_vector_store: PGVectorStore | None = None


async def init_vector_store() -> None:
    global _vector_store  # noqa: PLW0603
    _vector_store = await PGVectorStore.create(
        pg_engine,
        embedding_service=embedding,
        table_name=settings.postgres.embedding_table_name,
    )


def get_vector_store() -> PGVectorStore:
    if _vector_store is None:
        raise RuntimeError(VECTOR_STORE_NOT_INITIALIZED)
    return _vector_store
