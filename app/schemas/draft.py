from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.schemas.retrieval import (
    RetrievalEvidence,
)

from app.schemas.validation import (
    ValidationResult,
)


# ============================================================
# Types
# ============================================================


OutputLanguage = Literal[
    "ko",
    "en",
]


# ============================================================
# Request
# ============================================================


class DraftRequest(BaseModel):
    """
    POST /api/v1/draft 요청.

    최종적으로 Stage2 Transformer가 받을 서비스 입력 형태와
    최대한 비슷하게 유지한다.

    Title
    + Human Research Idea
    + Evidence
    + Constraints
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    title: str = Field(
        min_length=3,
        max_length=500,
        description="Research title",
    )

    research_idea: str = Field(
        min_length=3,
        max_length=3000,
        description="Human-provided original research idea",
    )

    writing_constraints: list[str] = Field(
        default_factory=list,
        description="Writing constraints supplied by the user",
    )

    language: OutputLanguage = Field(
        default="ko",
        description="Output language",
    )

    target_length: int = Field(
        default=4500,
        ge=1000,
        le=10000,
        description="Target output length in characters",
    )

    top_k: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of final reranked evidence items",
    )


# ============================================================
# Paper structure
# ============================================================


class PaperSection(BaseModel):
    """
    Structured paper section.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    heading: str = Field(
        min_length=1,
    )

    body: str


class ReferenceStub(BaseModel):
    """
    현재 2차 Backend 개발 단계의 임시 Reference 구조.

    실제 References는 추후:
        document_id
        -> DB metadata
        -> Backend bibliography assembly

    로 확장한다.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    document_id: int

    document_title: str

    source: str

    source_id: str


# ============================================================
# Generator metadata
# ============================================================


class GeneratorMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    generator_type: str

    generator_version: str

    is_mock: bool


# ============================================================
# Response
# ============================================================


class DraftResponse(BaseModel):
    """
    Frontend가 Paper Preview를 구성할 수 있는
    Structured Paper JSON.

    현재 MockGenerator를 사용하지만,
    나중에 OwnTransformerGenerator로 바뀌어도
    response contract는 그대로 유지한다.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    title: str

    abstract: str

    sections: list[
        PaperSection
    ]

    evidence: list[
        RetrievalEvidence
    ]

    references: list[
        ReferenceStub
    ]

    generator: GeneratorMetadata

    language: OutputLanguage

    target_length: int

    validation: ValidationResult | None = None