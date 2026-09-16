from functools import lru_cache

from psycopg.conninfo import make_conninfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.

    Values are loaded from the .env file.
    """

    # --------------------------------------------------
    # Application
    # --------------------------------------------------
    app_name: str = "Team B AI Paper Agent Backend"
    app_env: str = "development"
    api_prefix: str = "/api/v1"

    cors_origins: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173"
    )

    # --------------------------------------------------
    # PostgreSQL / AWS RDS
    # --------------------------------------------------
    db_host: str
    db_port: int = 5432
    db_name: str = "team_b_ai"
    db_user: str
    db_password: str
    db_sslmode: str = "require"
    db_connect_timeout: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def database_dsn(self) -> str:
        """
        Build a safe psycopg connection string.

        make_conninfo() safely handles special characters
        in usernames and passwords.
        """
        return make_conninfo(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            sslmode=self.db_sslmode,
            connect_timeout=self.db_connect_timeout,
        )

    @property
    def cors_origin_list(self) -> list[str]:
        """
        Convert comma-separated CORS origins to a list.
        """
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()