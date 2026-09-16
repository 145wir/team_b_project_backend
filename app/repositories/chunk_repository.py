from typing import Any

from app.core.database import get_connection


def _build_chunk_filters(
    *,
    document_id: int | None = None,
    source: str | None = None,
    topic_axis: str | None = None,
    section_heading: str | None = None,
) -> tuple[str, list[Any]]:
    """
    core_chunks + core_documents 조회에 사용할
    공통 WHERE 조건을 생성한다.
    """

    conditions: list[str] = []
    parameters: list[Any] = []

    if document_id is not None:
        conditions.append("c.document_id = %s")
        parameters.append(document_id)

    if source:
        conditions.append("d.source = %s")
        parameters.append(source)

    if topic_axis:
        conditions.append("d.topic_axis = %s")
        parameters.append(topic_axis)

    if section_heading:
        conditions.append("c.section_heading ILIKE %s")
        parameters.append(f"%{section_heading}%")

    if not conditions:
        return "", parameters

    return "WHERE " + " AND ".join(conditions), parameters


def list_chunks(
    *,
    document_id: int | None = None,
    source: str | None = None,
    topic_axis: str | None = None,
    section_heading: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """
    Chunk 목록 조회.

    목록 API에서는 전체 chunk_text를 그대로 반환하지 않고
    앞부분 500자만 chunk_preview로 반환한다.
    """

    where_clause, parameters = _build_chunk_filters(
        document_id=document_id,
        source=source,
        topic_axis=topic_axis,
        section_heading=section_heading,
    )

    query = f"""
        SELECT
            c.id,
            c.document_id,
            c.chunk_index,
            c.section_index,
            c.section_heading,

            LEFT(c.chunk_text, 500) AS chunk_preview,

            d.title AS document_title,
            d.source,
            d.source_id,
            d.topic_axis,
            d.url

        FROM public.core_chunks AS c

        JOIN public.core_documents AS d
          ON d.id = c.document_id

        {where_clause}

        ORDER BY
            c.document_id ASC,
            c.chunk_index ASC,
            c.id ASC

        LIMIT %s
        OFFSET %s
    """

    parameters.extend([limit, offset])

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()

    return list(rows)


def count_chunks(
    *,
    document_id: int | None = None,
    source: str | None = None,
    topic_axis: str | None = None,
    section_heading: str | None = None,
) -> int:
    """
    list_chunks와 동일한 조건으로 전체 개수를 계산한다.
    """

    where_clause, parameters = _build_chunk_filters(
        document_id=document_id,
        source=source,
        topic_axis=topic_axis,
        section_heading=section_heading,
    )

    query = f"""
        SELECT
            COUNT(*) AS count

        FROM public.core_chunks AS c

        JOIN public.core_documents AS d
          ON d.id = c.document_id

        {where_clause}
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            row = cursor.fetchone()

    if row is None:
        return 0

    return int(row["count"])


def get_chunk_by_id(
    chunk_id: int,
) -> dict | None:
    """
    Chunk 하나를 전체 본문과 함께 조회한다.

    c.*를 사용하므로 이후 vector 관련 컬럼이나
    word_count 등의 컬럼이 추가되어도 그대로 반환 가능하다.
    """

    query = """
        SELECT
            c.*,

            d.title AS document_title,
            d.source,
            d.source_id,
            d.topic_axis,
            d.url

        FROM public.core_chunks AS c

        JOIN public.core_documents AS d
          ON d.id = c.document_id

        WHERE c.id = %s

        LIMIT 1
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (chunk_id,),
            )

            row = cursor.fetchone()

    return row


def get_chunk_stats() -> dict:
    """
    현재 Chunk Corpus의 간단한 통계를 반환한다.

    기존 QA 결과와 Backend에서 보는 DB 상태가
    일치하는지 확인하는 용도로 사용한다.
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:

            # --------------------------------------------------
            # 기본 통계
            # --------------------------------------------------
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_chunks,

                    COUNT(
                        DISTINCT document_id
                    ) AS documents_covered,

                    COUNT(*) FILTER (
                        WHERE
                            chunk_text IS NULL
                            OR BTRIM(chunk_text) = ''
                    ) AS empty_chunks,

                    COALESCE(
                        AVG(
                            CHAR_LENGTH(chunk_text)
                        ),
                        0
                    ) AS average_characters,

                    COALESCE(
                        MAX(
                            CHAR_LENGTH(chunk_text)
                        ),
                        0
                    ) AS max_characters

                FROM public.core_chunks
                """
            )

            base_row = cursor.fetchone()

            # --------------------------------------------------
            # Source별
            # --------------------------------------------------
            cursor.execute(
                """
                SELECT
                    d.source,
                    COUNT(*) AS count

                FROM public.core_chunks AS c

                JOIN public.core_documents AS d
                  ON d.id = c.document_id

                GROUP BY d.source
                ORDER BY d.source
                """
            )

            source_rows = cursor.fetchall()

            # --------------------------------------------------
            # Topic Axis별
            # --------------------------------------------------
            cursor.execute(
                """
                SELECT
                    d.topic_axis,
                    COUNT(*) AS count

                FROM public.core_chunks AS c

                JOIN public.core_documents AS d
                  ON d.id = c.document_id

                GROUP BY d.topic_axis
                ORDER BY d.topic_axis
                """
            )

            topic_rows = cursor.fetchall()

    if base_row is None:
        return {
            "total_chunks": 0,
            "documents_covered": 0,
            "empty_chunks": 0,
            "average_characters": 0.0,
            "max_characters": 0,
            "by_source": {},
            "by_topic_axis": {},
        }

    return {
        "total_chunks": int(
            base_row["total_chunks"]
        ),
        "documents_covered": int(
            base_row["documents_covered"]
        ),
        "empty_chunks": int(
            base_row["empty_chunks"]
        ),
        "average_characters": round(
            float(
                base_row["average_characters"]
            ),
            2,
        ),
        "max_characters": int(
            base_row["max_characters"]
        ),
        "by_source": {
            row["source"]: int(
                row["count"]
            )
            for row in source_rows
        },
        "by_topic_axis": {
            row["topic_axis"]: int(
                row["count"]
            )
            for row in topic_rows
        },
    }