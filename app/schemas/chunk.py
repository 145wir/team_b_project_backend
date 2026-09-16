from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


TopicAxis = Literal[
    "rover_autonomy",
    "onboard_ai",
    "satellite_autonomy",
]


class ChunkSummary(BaseModel):
    """
    Chunk list용 가벼운 응답 모델.

    전체 chunk_text 대신 preview만 반환한다.
    """

    model_config = ConfigDict(
        extra="allow",
    )

    id: int
    document_id: int

    chunk_index: int

    section_index: int | None = None
    section_heading: str | None = None

    chunk_preview: str

    # core_documents JOIN 결과
    document_title: str | None = None
    source: str | None = None
    source_id: str | None = None
    topic_axis: TopicAxis | None = None
    url: str | None = None


class ChunkDetail(BaseModel):
    """
    Chunk 단건 상세 조회.

    실제 RAG가 사용할 chunk_text 전체를 반환한다.
    """

    model_config = ConfigDict(
        extra="allow",
    )

    id: int
    document_id: int

    chunk_index: int

    section_index: int | None = None
    section_heading: str | None = None

    chunk_text: str

    # 아래 필드는 DB에 존재할 경우 자동 포함 가능
    word_count: int | None = None
    estimated_token_count: int | None = None
    token_count: int | None = None

    # core_documents JOIN
    document_title: str | None = None
    source: str | None = None
    source_id: str | None = None
    topic_axis: TopicAxis | None = None
    url: str | None = None

    # 향후 vector/retrieval 관련 컬럼이 생겨도
    # extra="allow" 때문에 응답 모델을 당장 깨지 않는다.


class ChunkListResponse(BaseModel):
    total: int
    limit: int
    offset: int

    items: list[ChunkSummary]


class ChunkStatsResponse(BaseModel):
    """
    현재 Core Chunk Corpus 상태 확인용.
    """

    total_chunks: int
    documents_covered: int

    empty_chunks: int

    average_characters: float
    max_characters: int

    by_source: dict[str, int]
    by_topic_axis: dict[str, int]


class DocumentChunksResponse(BaseModel):
    """
    특정 논문의 모든 chunk 조회용.
    """

    document_id: int
    total: int
    limit: int
    offset: int

    items: list[ChunkSummary]