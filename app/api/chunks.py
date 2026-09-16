from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.repositories.chunk_repository import (
    count_chunks,
    get_chunk_by_id,
    get_chunk_stats,
    list_chunks,
)
from app.schemas.chunk import (
    ChunkDetail,
    ChunkListResponse,
    ChunkStatsResponse,
)


router = APIRouter(
    prefix="/chunks",
    tags=["Chunks"],
)


Source = Literal[
    "arxiv",
    "ntrs",
]

TopicAxis = Literal[
    "rover_autonomy",
    "onboard_ai",
    "satellite_autonomy",
]


@router.get(
    "",
    response_model=ChunkListResponse,
)
def read_chunks(
    document_id: int | None = Query(
        default=None,
        ge=1,
        description=(
            "특정 core_documents.id의 "
            "chunk만 조회"
        ),
    ),
    source: Source | None = Query(
        default=None,
        description=(
            "arxiv 또는 ntrs source filter"
        ),
    ),
    topic_axis: TopicAxis | None = Query(
        default=None,
        description=(
            "Team B 연구 주제축 filter"
        ),
    ),
    section_heading: str | None = Query(
        default=None,
        description=(
            "Section heading 부분 검색"
        ),
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> ChunkListResponse:
    """
    Core Chunk 목록 조회.

    목록에서는 전체 chunk_text 대신
    최대 500자의 preview를 반환한다.
    """

    chunks = list_chunks(
        document_id=document_id,
        source=source,
        topic_axis=topic_axis,
        section_heading=section_heading,
        limit=limit,
        offset=offset,
    )

    total = count_chunks(
        document_id=document_id,
        source=source,
        topic_axis=topic_axis,
        section_heading=section_heading,
    )

    return ChunkListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=chunks,
    )


@router.get(
    "/stats",
    response_model=ChunkStatsResponse,
)
def read_chunk_stats() -> ChunkStatsResponse:
    """
    Core Chunk Corpus 전체 통계.

    현재 100 documents / 3,113 chunks
    상태를 Backend에서도 확인할 수 있게 한다.
    """

    stats = get_chunk_stats()

    return ChunkStatsResponse(
        **stats,
    )


@router.get(
    "/{chunk_id}",
    response_model=ChunkDetail,
)
def read_chunk(
    chunk_id: int,
) -> ChunkDetail:
    """
    Chunk 단건 상세 조회.

    실제 chunk_text 전체와
    연결된 원 논문 metadata를 반환한다.
    """

    chunk = get_chunk_by_id(
        chunk_id
    )

    if chunk is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Chunk {chunk_id} "
                "was not found."
            ),
        )

    return ChunkDetail(
        **chunk
    )