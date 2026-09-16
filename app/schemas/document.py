from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


TopicAxis = Literal[
    "rover_autonomy",
    "onboard_ai",
    "satellite_autonomy",
]


class DocumentSummary(BaseModel):
    """
    Lightweight representation used in document lists.
    """

    model_config = ConfigDict(extra="ignore")

    id: int
    source: str
    source_id: str
    title: str

    abstract: str | None = None
    published_at: datetime | None = None

    document_type: str | None = None
    doi: str | None = None
    url: str | None = None

    topic_axis: TopicAxis
    language: str

    char_count: int | None = None
    parse_status: str | None = None
    normalization_version: str | None = None

    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentDetail(BaseModel):
    """
    Full representation of one Core document.
    """

    model_config = ConfigDict(extra="ignore")

    id: int
    source: str
    source_id: str
    title: str

    abstract: str | None = None

    authors: list[Any] | dict[str, Any] | None = None
    categories: list[Any] | dict[str, Any] | None = None

    published_at: datetime | None = None

    document_type: str | None = None

    doi: str | None = None
    url: str | None = None
    pdf_url: str | None = None

    topic_axis: TopicAxis
    language: str

    raw_content: str | None = None
    clean_content: str | None = None

    content_hash: str | None = None
    normalization_version: str | None = None

    char_count: int | None = None
    parse_status: str | None = None

    metadata: dict[str, Any] | list[Any] | None = None

    created_at: datetime | None = None
    updated_at: datetime | None = None


class DocumentListResponse(BaseModel):
    """
    Paginated document response.
    """

    total: int
    limit: int
    offset: int
    items: list[DocumentSummary]


class DocumentStatsResponse(BaseModel):
    """
    Corpus statistics response.
    """

    total_documents: int
    by_source: dict[str, int]
    by_topic_axis: dict[str, int]
    by_parse_status: dict[str, int]