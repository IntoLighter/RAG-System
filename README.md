# FastAPI RAG

A RAG service built with FastAPI, LangChain, PostgreSQL with pgvector, and Qwen models served with vLLM.

## Architecture

```text
                     ┌─────────────┐
                     │   FastAPI   │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌───────────┐ ┌───────────┐ ┌──────────┐
        │  Postgres │ │ Embedding │ │ Reranker │
        │  pgvector │ └───────────┘ └──────────┘
        │chat_hist. │
        └───────────┘       ┌─────────────┐
                            │ Qwen3-4B-AWQ│
                            └─────────────┘
```

Postgres stores two things: the document chunks with their embeddings in the
`embedding` pgvector table (`PGVectorStore`, HNSW index, cosine distance), and
the messages of every chat session in the `chat` table
(`PostgresChatMessageHistory`), so requests sharing a `session_id` keep their
history. Both tables (and the index) are created by Alembic migrations
(`apps/migrations/`), which the one-shot `migrations` compose service
applies with `alembic upgrade head` before the API starts; their names come
from `POSTGRES__EMBEDDING_TABLE_NAME` and `POSTGRES__CHAT_TABLE_NAME`.

## Stack

* Python 3.13+
* FastAPI
* LangChain
* langchain-postgres (`PGVectorStore`, v2 engine)
* PostgreSQL with pgvector
* psycopg (async driver)
* vLLM
* Docker Compose
* Qwen3-4B-AWQ
* Qwen3-Embedding-0.6B
* Qwen3-Reranker-0.6B

## RAG Pipeline

1. Documents are split into chunks.
2. Embeddings are generated and stored in PostgreSQL with pgvector.
3. The query retrieves relevant chunks.
4. Retrieved chunks are reranked.
5. The most relevant chunks are passed to the generative model.
6. The model generates the final answer.

## Run

Create a `.env` file and start the services:

```bash
docker compose up -d
```

Check the service status:

```bash
docker compose ps
```

Stop the services:

```bash
docker compose down
```

## Conversations

Each query accepts an optional `session_id`. Omit it to start a new
conversation; reuse the returned one to continue it:

```bash
curl -X POST http://localhost/query \
  -H 'Content-Type: application/json' \
  -d '{"message": "How do I deploy the service?"}'
```

```bash
curl -X POST http://localhost/query \
  -H 'Content-Type: application/json' \
  -d '{"message": "And in debug mode?", "session_id": "<session_id>"}'
```

The second call sees the previous messages: the user question and the final
answer of every turn are stored in Postgres through an async psycopg connection
pool and passed to the agent on the next request.

## Development

The repository is a [uv](https://docs.astral.sh/uv/) workspace with one
lockfile and two members, each with its own dependencies and Dockerfile:

* `apps/api` — the FastAPI service
* `apps/migrations` — the Alembic migrations

Install the dependencies of both:

```bash
uv sync --all-packages
```

Run linting and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

To drop the stored vectors and conversations, roll the migrations back and
apply them again (locally the host comes from `apps/migrations/.env`, since
`.env` points at the compose service):

```bash
uv run alembic downgrade base
uv run alembic upgrade head
```

Note: the psycopg async driver (used by both the vector store and the
chat history) cannot run on the Windows proactor event loop that the plain
`fastapi run` entry point forces, so start the API locally with `fastapi dev`
(or `uvicorn --reload`).
