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

vector_store: PGVectorStore = PGVectorStore.create_sync(
    pg_engine,
    embedding_service=embedding,
    table_name=settings.postgres.embedding_table_name,
)
