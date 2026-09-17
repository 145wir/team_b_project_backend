from functools import lru_cache
from pathlib import Path

from psycopg.conninfo import make_conninfo
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


BACKEND_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

ENV_FILE = (
    BACKEND_ROOT
    / ".env"
)


class Settings(BaseSettings):

    # ========================================================
    # Application
    # ========================================================

    app_name: str = (
        "Team B AI Paper Agent Backend"
    )

    app_env: str = "development"

    api_prefix: str = "/api/v1"

    cors_origins: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173"
    )


    # ========================================================
    # AWS RDS PostgreSQL
    # ========================================================

    db_host: str

    db_port: int = 5432

    db_name: str = "team_b_ai"

    db_user: str

    db_password: str

    db_sslmode: str = "require"

    db_connect_timeout: int = 10

    # ========================================================
    # Retrieval
    # ========================================================

    retriever_mode: str = "mock"

    retrieval_candidate_k: int = 20

    retrieval_top_k: int = 6


    # ========================================================
    # Pydantic Settings
    # ========================================================

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


    @property
    def database_dsn(
        self,
    ) -> str:
        return make_conninfo(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            sslmode=self.db_sslmode,
            connect_timeout=(
                self.db_connect_timeout
            ),
        )


    @property
    def cors_origin_list(
        self,
    ) -> list[str]:
        return [
            origin.strip()

            for origin
            in self.cors_origins.split(
                ","
            )

            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()