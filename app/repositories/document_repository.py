from typing import Any

from app.core.database import get_connection


SUMMARY_COLUMNS = """
    id,
    source,
    source_id,
    title,
    abstract,
    published_at,
    document_type,
    doi,
    url,
    topic_axis,
    language,
    char_count,
    parse_status,
    normalization_version,
    created_at,
    updated_at
"""


DETAIL_COLUMNS = """
    id,
    source,
    source_id,
    title,
    abstract,
    authors,
    categories,
    published_at,
    document_type,
    doi,
    url,
    pdf_url,
    topic_axis,
    language,
    raw_content,
    clean_content,
    content_hash,
    normalization_version,
    char_count,
    parse_status,
    metadata,
    created_at,
    updated_at
"""


def _build_filters(
    source: str | None = None,
    topic_axis: str | None = None,
    parse_status: str | None = None,
) -> tuple[str, list[Any]]:
    """
    Build WHERE conditions using parameterized values.

    Column names are not supplied by the user, so this avoids
    SQL injection while keeping optional filters simple.
    """

    conditions: list[str] = []
    parameters: list[Any] = []

    if source:
        conditions.append("source = %s")
        parameters.append(source)

    if topic_axis:
        conditions.append("topic_axis = %s")
        parameters.append(topic_axis)

    if parse_status:
        conditions.append("parse_status = %s")
        parameters.append(parse_status)

    if not conditions:
        return "", parameters

    return "WHERE " + " AND ".join(conditions), parameters


def list_documents(
    *,
    source: str | None = None,
    topic_axis: str | None = None,
    parse_status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """
    Return a paginated list of Core documents.

    raw_content and clean_content are intentionally excluded
    from list responses because they can be very large.
    """

    where_clause, parameters = _build_filters(
        source=source,
        topic_axis=topic_axis,
        parse_status=parse_status,
    )

    query = f"""
        SELECT
            {SUMMARY_COLUMNS}
        FROM public.core_documents
        {where_clause}
        ORDER BY id DESC
        LIMIT %s
        OFFSET %s
    """

    parameters.extend([limit, offset])

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()

    return list(rows)


def count_documents(
    *,
    source: str | None = None,
    topic_axis: str | None = None,
    parse_status: str | None = None,
) -> int:
    """
    Count Core documents using the same filters as list_documents().
    """

    where_clause, parameters = _build_filters(
        source=source,
        topic_axis=topic_axis,
        parse_status=parse_status,
    )

    query = f"""
        SELECT COUNT(*) AS count
        FROM public.core_documents
        {where_clause}
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            row = cursor.fetchone()

    if row is None:
        return 0

    return int(row["count"])


def get_document_by_id(document_id: int) -> dict | None:
    """
    Return a single document including raw/clean full text.
    """

    query = f"""
        SELECT
            {DETAIL_COLUMNS}
        FROM public.core_documents
        WHERE id = %s
        LIMIT 1
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (document_id,))
            row = cursor.fetchone()

    return row


def get_document_stats() -> dict:
    """
    Return simple project statistics.

    This endpoint will be useful during development and presentation
    because the team can immediately verify the amount and balance
    of the Core Corpus.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) AS total
                FROM public.core_documents
                """
            )
            total_row = cursor.fetchone()

            cursor.execute(
                """
                SELECT
                    source,
                    COUNT(*) AS count
                FROM public.core_documents
                GROUP BY source
                ORDER BY source
                """
            )
            source_rows = cursor.fetchall()

            cursor.execute(
                """
                SELECT
                    topic_axis,
                    COUNT(*) AS count
                FROM public.core_documents
                GROUP BY topic_axis
                ORDER BY topic_axis
                """
            )
            topic_rows = cursor.fetchall()

            cursor.execute(
                """
                SELECT
                    parse_status,
                    COUNT(*) AS count
                FROM public.core_documents
                GROUP BY parse_status
                ORDER BY parse_status
                """
            )
            status_rows = cursor.fetchall()

    total = 0

    if total_row is not None:
        total = int(total_row["total"])

    return {
        "total_documents": total,
        "by_source": {
            row["source"]: int(row["count"])
            for row in source_rows
        },
        "by_topic_axis": {
            row["topic_axis"]: int(row["count"])
            for row in topic_rows
        },
        "by_parse_status": {
            row["parse_status"]: int(row["count"])
            for row in status_rows
        },
    }