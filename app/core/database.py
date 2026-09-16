from collections.abc import Generator
from contextlib import contextmanager

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from app.core.config import settings


@contextmanager
def get_connection() -> Generator[Connection, None, None]:
    """
    Open a PostgreSQL connection and close it automatically.

    A new connection is created for each repository operation
    in this first backend version.

    Later, if traffic increases, this can be replaced by
    psycopg connection pooling without changing the API layer.
    """

    connection = psycopg.connect(
        settings.database_dsn,
        row_factory=dict_row,
    )

    try:
        yield connection
    finally:
        connection.close()


def check_database_connection() -> dict:
    """
    Verify that the backend can connect to AWS RDS.

    Returns basic database information without exposing
    credentials or sensitive connection details.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    current_database() AS database_name,
                    current_schema() AS schema_name,
                    version() AS postgres_version
                """
            )

            row = cursor.fetchone()

            if row is None:
                raise RuntimeError("Database health query returned no result.")

            return row