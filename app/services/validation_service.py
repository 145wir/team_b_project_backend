from __future__ import annotations

import re
from collections import Counter

from app.schemas.draft import (
    PaperSection,
)
from app.schemas.retrieval import (
    RetrievalEvidence,
)
from app.schemas.validation import (
    ValidationIssue,
    ValidationResult,
    ValidationSummary,
)


VALIDATOR_VERSION = (
    "rule_based_validator_v1"
)


SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?。！？])\s+"
)

NUMBER_PATTERN = re.compile(
    r"(?<!\w)"
    r"\d+(?:\.\d+)?"
    r"(?:\s*[%％])?"
    r"(?!\w)"
)

WORD_PATTERN = re.compile(
    r"[A-Za-z0-9가-힣]+"
)


# ============================================================
# Helpers
# ============================================================


def _normalize_text(
    text: str,
) -> str:
    return " ".join(
        str(
            text or ""
        ).split()
    )


def _split_sentences(
    text: str,
) -> list[str]:

    normalized = (
        _normalize_text(
            text
        )
    )

    if not normalized:
        return []

    sentences = (
        SENTENCE_SPLIT_PATTERN
        .split(
            normalized
        )
    )

    return [
        sentence.strip()

        for sentence
        in sentences

        if sentence.strip()
    ]


def _tokenize_words(
    text: str,
) -> list[str]:

    return [
        token.lower()

        for token
        in WORD_PATTERN.findall(
            text or ""
        )
    ]


def _extract_numbers(
    text: str,
) -> set[str]:

    normalized = str(
        text or ""
    )

    # 개발용 Evidence label 및 citation marker의
    # 숫자는 scientific numerical claim으로 보지 않는다.
    normalized = re.sub(
        r"\[\s*Evidence\s+\d+\s*\]",
        "",
        normalized,
        flags=re.IGNORECASE,
    )

    return {
        match.group(0)
        .replace(
            " ",
            "",
        )

        for match
        in NUMBER_PATTERN.finditer(
            normalized
        )
    }


def _make_ngrams(
    tokens: list[str],
    size: int,
) -> set[
    tuple[str, ...]
]:

    if (
        size <= 0
        or len(
            tokens
        )
        < size
    ):
        return set()

    return {
        tuple(
            tokens[
                index:
                index + size
            ]
        )

        for index
        in range(
            len(tokens)
            - size
            + 1
        )
    }


# ============================================================
# Structure
# ============================================================


def _check_structure(
    sections: list[
        PaperSection
    ],
) -> list[
    ValidationIssue
]:

    issues: list[
        ValidationIssue
    ] = []

    headings = {
        section.heading
        .strip()
        .lower()

        for section
        in sections
    }

    expected = {
        "introduction",
        "body",
        "conclusion",
    }

    missing = (
        expected
        - headings
    )

    for heading in sorted(
        missing
    ):
        issues.append(
            ValidationIssue(
                issue_type="structure",
                severity="error",
                section=heading,
                message=(
                    f"Required section "
                    f"'{heading}' is missing."
                ),
            )
        )

    for section in sections:
        if not section.body.strip():
            issues.append(
                ValidationIssue(
                    issue_type="structure",
                    severity="error",
                    section=(
                        section.heading
                    ),
                    message=(
                        "Section body is empty."
                    ),
                )
            )

    return issues


# ============================================================
# Repetition
# ============================================================


def _check_repetition(
    sections: list[
        PaperSection
    ],
) -> list[
    ValidationIssue
]:

    issues: list[
        ValidationIssue
    ] = []

    for section in sections:

        sentences = (
            _split_sentences(
                section.body
            )
        )

        normalized = [
            sentence.lower()

            for sentence
            in sentences

            if len(
                sentence
            )
            >= 20
        ]

        counts = Counter(
            normalized
        )

        for (
            sentence,
            count,
        ) in counts.items():

            if count <= 1:
                continue

            issues.append(
                ValidationIssue(
                    issue_type="repetition",
                    severity="warning",
                    section=(
                        section.heading
                    ),
                    message=(
                        "Repeated sentence "
                        f"detected {count} times."
                    ),
                    matched_text=sentence,
                )
            )

    return issues


# ============================================================
# Unsupported Numbers
# ============================================================


def _check_unsupported_numbers(
    sections: list[
        PaperSection
    ],
    evidence: list[
        RetrievalEvidence
    ],
) -> list[
    ValidationIssue
]:

    issues: list[
        ValidationIssue
    ] = []

    evidence_numbers: set[str] = set()

    for item in evidence:
        evidence_numbers.update(
            _extract_numbers(
                item.chunk_text
            )
        )

    for section in sections:

        generated_numbers = (
            _extract_numbers(
                section.body
            )
        )

        unsupported = (
            generated_numbers
            - evidence_numbers
        )

        for number in sorted(
            unsupported
        ):
            issues.append(
                ValidationIssue(
                    issue_type=(
                        "unsupported_number"
                    ),
                    severity="warning",
                    section=(
                        section.heading
                    ),
                    message=(
                        "A numeric claim was "
                        "not found in retrieved "
                        "evidence."
                    ),
                    matched_text=number,
                )
            )

    return issues


# ============================================================
# Exact Copy
# ============================================================


def _check_exact_copy(
    sections: list[
        PaperSection
    ],
    evidence: list[
        RetrievalEvidence
    ],
) -> list[
    ValidationIssue
]:

    issues: list[
        ValidationIssue
    ] = []

    normalized_evidence = [
        (
            item.rank,
            _normalize_text(
                item.chunk_text
            ).lower(),
        )

        for item
        in evidence
    ]

    for section in sections:

        sentences = (
            _split_sentences(
                section.body
            )
        )

        for sentence in sentences:

            normalized_sentence = (
                _normalize_text(
                    sentence
                ).lower()
            )

            # 지나치게 짧은 일치 문장은
            # 표절 신호로 사용하지 않는다.
            if (
                len(
                    normalized_sentence
                )
                < 80
            ):
                continue

            for (
                evidence_rank,
                evidence_text,
            ) in normalized_evidence:

                if (
                    normalized_sentence
                    in evidence_text
                ):
                    issues.append(
                        ValidationIssue(
                            issue_type=(
                                "exact_copy"
                            ),
                            severity=(
                                "warning"
                            ),
                            section=(
                                section.heading
                            ),
                            message=(
                                "A long generated "
                                "sentence exactly "
                                "matches retrieved "
                                "evidence."
                            ),
                            evidence_rank=(
                                evidence_rank
                            ),
                            matched_text=(
                                sentence
                            ),
                        )
                    )

                    break

    return issues


# ============================================================
# N-gram overlap
# ============================================================


def _check_ngram_overlap(
    sections: list[
        PaperSection
    ],
    evidence: list[
        RetrievalEvidence
    ],
    ngram_size: int = 12,
) -> list[
    ValidationIssue
]:

    issues: list[
        ValidationIssue
    ] = []

    evidence_ngrams: list[
        tuple[
            int,
            set[
                tuple[str, ...]
            ],
        ]
    ] = []

    for item in evidence:

        tokens = _tokenize_words(
            item.chunk_text
        )

        evidence_ngrams.append(
            (
                item.rank,
                _make_ngrams(
                    tokens,
                    ngram_size,
                ),
            )
        )

    for section in sections:

        section_tokens = (
            _tokenize_words(
                section.body
            )
        )

        section_ngrams = (
            _make_ngrams(
                section_tokens,
                ngram_size,
            )
        )

        if not section_ngrams:
            continue

        for (
            evidence_rank,
            reference_ngrams,
        ) in evidence_ngrams:

            overlap = (
                section_ngrams
                & reference_ngrams
            )

            if not overlap:
                continue

            example = (
                " ".join(
                    next(
                        iter(
                            overlap
                        )
                    )
                )
            )

            issues.append(
                ValidationIssue(
                    issue_type=(
                        "ngram_overlap"
                    ),
                    severity="info",
                    section=(
                        section.heading
                    ),
                    message=(
                        f"{ngram_size}-token "
                        "overlap detected with "
                        "retrieved evidence."
                    ),
                    evidence_rank=(
                        evidence_rank
                    ),
                    matched_text=(
                        example
                    ),
                )
            )

    return issues


# ============================================================
# Public Validator
# ============================================================


def validate_draft(
    sections: list[
        PaperSection
    ],
    evidence: list[
        RetrievalEvidence
    ],
) -> ValidationResult:

    issues: list[
        ValidationIssue
    ] = []

    issues.extend(
        _check_structure(
            sections
        )
    )

    issues.extend(
        _check_repetition(
            sections
        )
    )

    issues.extend(
        _check_unsupported_numbers(
            sections,
            evidence,
        )
    )

    issues.extend(
        _check_exact_copy(
            sections,
            evidence,
        )
    )

    issues.extend(
        _check_ngram_overlap(
            sections,
            evidence,
        )
    )

    errors = sum(
        issue.severity
        == "error"

        for issue
        in issues
    )

    warnings = sum(
        issue.severity
        == "warning"

        for issue
        in issues
    )

    infos = sum(
        issue.severity
        == "info"

        for issue
        in issues
    )

    summary = ValidationSummary(
        passed=(
            errors == 0
        ),

        total_issues=(
            len(
                issues
            )
        ),

        errors=errors,

        warnings=warnings,

        infos=infos,
    )

    return ValidationResult(
        validator_version=(
            VALIDATOR_VERSION
        ),

        summary=summary,

        issues=issues,
    )