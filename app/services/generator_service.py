from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from app.schemas.draft import (
    DraftRequest,
    DraftResponse,
    GeneratorMetadata,
    PaperSection,
    ReferenceStub,
)
from app.schemas.retrieval import (
    RetrievalResponse,
)


# ============================================================
# Version
# ============================================================


MOCK_GENERATOR_VERSION = (
    "mock_generator_v1"
)


# ============================================================
# Base Generator
# ============================================================


class BaseGenerator(ABC):
    """
    모든 Generator가 따라야 하는 interface.

    현재:
        MockGenerator

    향후:
        OwnTransformerGenerator

    로 교체한다.
    """

    @abstractmethod
    def generate(
        self,
        request: DraftRequest,
        retrieval: RetrievalResponse,
    ) -> DraftResponse:
        raise NotImplementedError


# ============================================================
# Helpers
# ============================================================


def _truncate_text(
    text: str,
    limit: int,
) -> str:
    normalized = (
        " ".join(
            str(
                text or ""
            ).split()
        )
    )

    if (
        len(
            normalized
        )
        <= limit
    ):
        return normalized

    return (
        normalized[
            :limit
        ].rstrip()
        + "..."
    )


def _build_reference_stubs(
    retrieval: RetrievalResponse,
) -> list[
    ReferenceStub
]:
    """
    같은 논문에서 여러 chunk가 선택돼도
    References에서는 document 단위로 하나만 만든다.

    현재는 bibliography 완성 전이므로 stub만 생성한다.
    """

    references: list[
        ReferenceStub
    ] = []

    seen_document_ids: set[
        int
    ] = set()

    for evidence in retrieval.evidence:
        if (
            evidence.document_id
            in seen_document_ids
        ):
            continue

        seen_document_ids.add(
            evidence.document_id
        )

        references.append(
            ReferenceStub(
                document_id=(
                    evidence.document_id
                ),
                document_title=(
                    evidence.document_title
                ),
                source=(
                    evidence.source
                ),
                source_id=(
                    evidence.source_id
                ),
            )
        )

    return references


def _group_evidence_text(
    retrieval: RetrievalResponse,
    max_items: int = 3,
) -> str:
    """
    Mock output이 없는 사실을 만들지 않도록
    실제 Evidence 일부를 개발용 preview에 사용한다.
    """

    excerpts: list[str] = []

    for evidence in (
        retrieval.evidence[
            :max_items
        ]
    ):
        excerpt = _truncate_text(
            evidence.chunk_text,
            280,
        )

        excerpts.append(
            (
                f"[Evidence {evidence.rank}] "
                f"{excerpt}"
            )
        )

    return "\n\n".join(
        excerpts
    )


# ============================================================
# Mock Generator
# ============================================================


class MockGenerator(
    BaseGenerator
):
    """
    실제 Transformer checkpoint가 준비될 때까지
    Backend / Frontend E2E를 검증하기 위한 Generator.

    중요한 원칙:
    - 새로운 연구 결과를 만들어내지 않는다.
    - 임의 숫자를 만들지 않는다.
    - 실제 논문이라고 가장하지 않는다.
    - Evidence 일부를 이용해 preview 구조만 만든다.
    """

    def generate(
        self,
        request: DraftRequest,
        retrieval: RetrievalResponse,
    ) -> DraftResponse:

        if request.language == "ko":
            return self._generate_korean(
                request=request,
                retrieval=retrieval,
            )

        return self._generate_english(
            request=request,
            retrieval=retrieval,
        )


    # ========================================================
    # Korean Mock
    # ========================================================

    def _generate_korean(
        self,
        request: DraftRequest,
        retrieval: RetrievalResponse,
    ) -> DraftResponse:

        evidence_preview = (
            _group_evidence_text(
                retrieval
            )
        )

        constraints_text = (
            ", ".join(
                request.writing_constraints
            )
            if request.writing_constraints
            else "별도 조건 없음"
        )

        abstract = (
            "[개발용 Mock Draft] "
            "본 초안은 실제 Transformer 생성 결과가 아니라 "
            "Backend와 Frontend의 End-to-End 연결을 검증하기 위한 "
            "임시 결과입니다. "
            f"연구 주제는 '{request.title}'이며, "
            "사용자가 제공한 연구 아이디어와 "
            f"{len(retrieval.evidence)}개의 검색 근거를 "
            "구조화하는 형태로 구성되었습니다."
        )

        introduction = (
            "[Mock Introduction]\n\n"
            f"연구 제목: {request.title}\n\n"
            f"사람이 제시한 연구 아이디어: "
            f"{request.research_idea}\n\n"
            "본 단계에서는 실제 Transformer 대신 "
            "MockGenerator를 사용하고 있습니다. "
            "따라서 새로운 과학적 주장이나 실험 결과를 "
            "생성하지 않으며, 최종 Reranker가 선택한 "
            "Evidence가 이후 Stage2 Transformer의 근거로 "
            "사용된다는 흐름만 검증합니다."
        )

        body = (
            "[Mock Body]\n\n"
            "현재 Retriever와 Reranker가 선택한 "
            "근거 중 일부는 다음과 같습니다.\n\n"
            f"{evidence_preview}\n\n"
            "실제 Stage2에서는 이러한 Evidence와 "
            "Human Research Idea를 함께 conditioning하여 "
            "새로운 학술 문장으로 조직하게 됩니다."
        )

        conclusion = (
            "[Mock Conclusion]\n\n"
            "현재 결과는 논문 품질 평가 대상이 아니라 "
            "Backend orchestration 검증용입니다. "
            "실제 Transformer 연결 후에는 "
            "Introduction / Body / Conclusion을 section 단위로 "
            "생성하고 Validator를 통해 grounding, "
            "unsupported numbers, citation, copy overlap, "
            "repetition 등을 검사할 예정입니다.\n\n"
            f"현재 적용된 작성 조건: {constraints_text}"
        )

        return DraftResponse(
            title=request.title,

            abstract=abstract,

            sections=[
                PaperSection(
                    heading="Introduction",
                    body=introduction,
                ),
                PaperSection(
                    heading="Body",
                    body=body,
                ),
                PaperSection(
                    heading="Conclusion",
                    body=conclusion,
                ),
            ],

            evidence=(
                retrieval.evidence
            ),

            references=(
                _build_reference_stubs(
                    retrieval
                )
            ),

            generator=(
                GeneratorMetadata(
                    generator_type=(
                        "MockGenerator"
                    ),
                    generator_version=(
                        MOCK_GENERATOR_VERSION
                    ),
                    is_mock=True,
                )
            ),

            language=request.language,

            target_length=(
                request.target_length
            ),
        )


    # ========================================================
    # English Mock
    # ========================================================

    def _generate_english(
        self,
        request: DraftRequest,
        retrieval: RetrievalResponse,
    ) -> DraftResponse:

        evidence_preview = (
            _group_evidence_text(
                retrieval
            )
        )

        constraints_text = (
            ", ".join(
                request.writing_constraints
            )
            if request.writing_constraints
            else "None"
        )

        abstract = (
            "[Development Mock Draft] "
            "This is not an output produced by the final "
            "Transformer model. It is a temporary structured "
            "response used to validate the Backend and Frontend "
            "end-to-end pipeline."
        )

        introduction = (
            "[Mock Introduction]\n\n"
            f"Research title: {request.title}\n\n"
            f"Human research idea: "
            f"{request.research_idea}\n\n"
            "The final system will condition the directly trained "
            "Transformer on the human idea and reranked evidence."
        )

        body = (
            "[Mock Body]\n\n"
            "Selected evidence preview:\n\n"
            f"{evidence_preview}\n\n"
            "The Stage2 Transformer will later organize these "
            "sources into scientific writing without inventing "
            "unsupported experimental results."
        )

        conclusion = (
            "[Mock Conclusion]\n\n"
            "This output validates orchestration only. "
            "The production pipeline will replace this generator "
            "with the team's own Transformer and subsequently "
            "apply the Validator.\n\n"
            f"Writing constraints: {constraints_text}"
        )

        return DraftResponse(
            title=request.title,

            abstract=abstract,

            sections=[
                PaperSection(
                    heading="Introduction",
                    body=introduction,
                ),
                PaperSection(
                    heading="Body",
                    body=body,
                ),
                PaperSection(
                    heading="Conclusion",
                    body=conclusion,
                ),
            ],

            evidence=(
                retrieval.evidence
            ),

            references=(
                _build_reference_stubs(
                    retrieval
                )
            ),

            generator=(
                GeneratorMetadata(
                    generator_type=(
                        "MockGenerator"
                    ),
                    generator_version=(
                        MOCK_GENERATOR_VERSION
                    ),
                    is_mock=True,
                )
            ),

            language=request.language,

            target_length=(
                request.target_length
            ),
        )


# ============================================================
# Generator factory
# ============================================================


def get_generator() -> BaseGenerator:
    """
    현재는 항상 MockGenerator.

    나중에는 환경설정 등을 통해:

        mock
        transformer

    를 선택하도록 변경할 수 있다.
    """

    return MockGenerator()