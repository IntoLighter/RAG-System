import sys
from pathlib import Path

import psycopg
from psycopg import sql

sys.path.append(str(Path(__file__).resolve().parents[1]))

from settings import settings

# The app recreates both tables on startup.
with psycopg.connect(settings.postgres.conninfo) as conn:
    for table in (
        settings.postgres.embedding_table_name,
        settings.postgres.chat_table_name,
    ):
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table)))
