from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

here = Path(__file__).resolve().parent


class PostgresSettings(BaseModel):
    host: str
    port: int
    database: str
    user: str
    password: str


class Settings(BaseSettings):
    # Later env files override earlier ones and real environment variables
    # override both, so the priority is:
    # shell environment > apps/migrations/.env > root .env
    # In the container neither file exists and compose passes the variables.
    model_config = SettingsConfigDict(
        env_file=(here.parent.parent / ".env", here / ".env"),
        env_nested_delimiter="__",
        extra="ignore",
    )

    postgres: PostgresSettings


settings = Settings()
