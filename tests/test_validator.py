from __future__ import annotations

from app.schemas.draft import (
    PaperSection,
)
from app.schemas.retrieval import (
    RetrievalEvidence,
)
from app.services.validation_service import (
    validate_draft,
)


# ============================================================
# Helpers
# ============================================================


def make_evidence(
    *,
    rank: int,
    chunk_id: int,
    document_id: int,
    chunk_text: str,
) -> RetrievalEvidence:
    return RetrievalEvidence(
        rank=rank,
        retriever_rank=rank,

        chunk_id=chunk_id,
        document_id=document_id,

        chunk_index=0,
        section_index=1,
        section_heading="Introduction",

        chunk_text=chunk_text,

        document_title=(
            f"Test Document {document_id}"
        ),

        source="ntrs",
        source_id=f"TEST-{document_id}",

        topic_axis="onboard_ai",

        cosine_similarity=0.70,
        quality_score=0.95,

        semantic_score=0.90,
        lexical_score=0.40,
        title_overlap_score=0.50,

        diversity_penalty=0.0,
        rerank_score=0.80,
    )


def make_evidence_list() -> list[
    RetrievalEvidence
]:
    return [
        make_evidence(
            rank=1,
            chunk_id=1,
            document_id=101,
            chunk_text=(
                "Deep-space spacecraft require autonomous "
                "decision-making because communication delays "
                "can prevent immediate intervention from "
                "ground operators."
            ),
        ),

        make_evidence(
            rank=2,
            chunk_id=2,
            document_id=102,
            chunk_text=(
                "Onboard artificial intelligence can support "
                "spacecraft monitoring, planning, and autonomous "
                "operations when continuous communication with "
                "Earth is unavailable."
            ),
        ),
    ]


def make_valid_sections() -> list[
    PaperSection
]:
    return [
        PaperSection(
            heading="Introduction",
            body=(
                "심우주 임무에서는 통신 지연으로 인해 "
                "지상 운영자가 모든 상황에 즉시 대응하기 어렵다. "
                "따라서 우주선이 일정 수준의 자율 판단 능력을 "
                "가지는 것이 중요한 연구 과제가 된다."
            ),
        ),

        PaperSection(
            heading="Body",
            body=(
                "검색된 근거들은 온보드 인공지능과 "
                "자율 의사결정이 우주선 운영에 활용될 수 있음을 "
                "보여준다. 이러한 근거를 바탕으로 인간의 감독과 "
                "자율성 사이의 균형을 연구할 수 있다."
            ),
        ),

        PaperSection(
            heading="Conclusion",
            body=(
                "심우주 자율화는 단순 자동화보다 넓은 개념이며, "
                "통신 제약을 고려한 의사결정 구조가 필요하다."
            ),
        ),
    ]


# ============================================================
# Test 1
# Normal draft
# ============================================================


def test_validator_accepts_well_formed_draft() -> None:
    """
    기본적인 Introduction / Body / Conclusion 구조가 있고
    특별한 오류가 없는 초안은 error 없이 검사되어야 한다.
    """

    result = validate_draft(
        sections=(
            make_valid_sections()
        ),
        evidence=(
            make_evidence_list()
        ),
    )

    assert (
        result.validator_version
        == "rule_based_validator_v1"
    )

    assert (
        result.summary.errors
        == 0
    )

    assert (
        result.summary.passed
        is True
    )


# ============================================================
# Test 2
# Unsupported numeric claim
# ============================================================


def test_validator_detects_unsupported_number() -> None:
    """
    Evidence에 없는 99.987%를 생성문에 추가했을 때
    unsupported_number warning이 발생해야 한다.
    """

    sections = (
        make_valid_sections()
    )

    sections[0].body += (
        " 실험 정확도는 99.987%로 측정되었다."
    )

    result = validate_draft(
        sections=sections,
        evidence=(
            make_evidence_list()
        ),
    )

    unsupported_issues = [
        issue
        for issue
        in result.issues
        if (
            issue.issue_type
            == "unsupported_number"
        )
    ]

    assert unsupported_issues

    assert any(
        "99.987"
        in (
            issue.matched_text
            or ""
        )
        for issue
        in unsupported_issues
    )


# ============================================================
# Test 3
# Evidence labels must not become numeric claims
# ============================================================


def test_validator_ignores_evidence_label_numbers() -> None:
    """
    [Evidence 1], [Evidence 2]의 숫자는
    scientific numeric claim으로 오인하면 안 된다.
    """

    sections = (
        make_valid_sections()
    )

    sections[1].body += (
        " [Evidence 1]은 첫 번째 근거이며, "
        "[Evidence 2]는 두 번째 근거이다."
    )

    result = validate_draft(
        sections=sections,
        evidence=(
            make_evidence_list()
        ),
    )

    unsupported_numbers = {
        issue.matched_text
        for issue
        in result.issues
        if (
            issue.issue_type
            == "unsupported_number"
        )
    }

    assert "1" not in (
        unsupported_numbers
    )

    assert "2" not in (
        unsupported_numbers
    )


# ============================================================
# Test 4
# Copy / n-gram overlap
# ============================================================


def test_validator_detects_long_evidence_overlap() -> None:
    """
    Evidence 문장을 생성문에 거의 그대로 복사하면
    exact_copy 또는 ngram_overlap이 잡혀야 한다.
    """

    evidence = (
        make_evidence_list()
    )

    copied_sentence = (
        evidence[0]
        .chunk_text
    )

    sections = (
        make_valid_sections()
    )

    sections[1].body = (
        copied_sentence
    )

    result = validate_draft(
        sections=sections,
        evidence=evidence,
    )

    overlap_types = {
        issue.issue_type
        for issue
        in result.issues
    }

    assert (
        "exact_copy"
        in overlap_types
        or
        "ngram_overlap"
        in overlap_types
    )


# ============================================================
# Test 5
# Required structure
# ============================================================


def test_validator_detects_missing_conclusion() -> None:
    """
    Conclusion section이 없으면
    structure error가 발생해야 한다.
    """

    sections = [
        PaperSection(
            heading="Introduction",
            body="Introduction text.",
        ),

        PaperSection(
            heading="Body",
            body="Body text.",
        ),
    ]

    result = validate_draft(
        sections=sections,
        evidence=(
            make_evidence_list()
        ),
    )

    structure_errors = [
        issue
        for issue
        in result.issues
        if (
            issue.issue_type
            == "structure"
            and
            issue.severity
            == "error"
        )
    ]

    assert structure_errors

    assert (
        result.summary.errors
        >= 1
    )

    assert (
        result.summary.passed
        is False
    )