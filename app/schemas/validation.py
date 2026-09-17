from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


ValidationSeverity = Literal[
    "info",
    "warning",
    "error",
]


ValidationType = Literal[
    "structure",
    "repetition",
    "unsupported_number",
    "exact_copy",
    "ngram_overlap",
]


class ValidationIssue(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
    )

    issue_type: ValidationType

    severity: ValidationSeverity

    section: str | None = None

    message: str

    evidence_rank: int | None = Field(
        default=None,
        ge=1,
    )

    matched_text: str | None = None


class ValidationSummary(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
    )

    passed: bool

    total_issues: int = Field(
        ge=0,
    )

    errors: int = Field(
        ge=0,
    )

    warnings: int = Field(
        ge=0,
    )

    infos: int = Field(
        ge=0,
    )


class ValidationResult(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
    )

    validator_version: str

    summary: ValidationSummary

    issues: list[
        ValidationIssue
    ]