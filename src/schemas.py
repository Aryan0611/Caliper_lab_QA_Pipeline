"""
Pydantic schemas for the 10-K Q&A Pipeline.
All data flowing through the pipeline is validated against these models.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class QuestionType(str, Enum):
    FACT_EXTRACTION       = "fact_extraction"
    NUMERIC_CALCULATION   = "numeric_calculation"
    COMPARISON            = "comparison"
    MULTI_STEP_REASONING  = "multi_step_reasoning"


class DifficultyLevel(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class ContentType(str, Enum):
    TEXT_PARAGRAPH  = "text_paragraph"
    FINANCIAL_TABLE = "financial_table"
    FOOTNOTE        = "footnote"
    MIXED           = "mixed"


class VerificationStatus(str, Enum):
    PASS   = "pass"
    FAIL   = "fail"
    REVISE = "revise"


class Chunk(BaseModel):
    chunk_id:               str
    content:                str
    item:                   str
    item_title:             str
    subsection:             Optional[str]        = None
    content_type:           ContentType          = ContentType.TEXT_PARAGRAPH
    token_count:            int                  = 0
    char_count:             int                  = 0
    contains_numbers:       bool                 = False
    fiscal_years_mentioned: List[str]            = Field(default_factory=list)
    parent_chunk_id:        Optional[str]        = None

    model_config = {"use_enum_values": True}


class RawQAPair(BaseModel):
    """Output from the generator LLM — not yet verified."""
    question:       str
    answer:         str
    source_passage: str
    reasoning:      Optional[str] = None

    @field_validator("question")
    @classmethod
    def ensure_question_mark(cls, v: str) -> str:
        v = v.strip()
        if not v.endswith("?"):
            v += "?"
        return v

    @field_validator("source_passage")
    @classmethod
    def source_not_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 10:
            raise ValueError("source_passage must contain meaningful text (min 10 chars)")
        return v.strip()


class QAPair(BaseModel):
    id:                  str
    question:            str
    ground_truth_answer: str
    source_passage:      str
    question_type:       QuestionType
    difficulty:          DifficultyLevel
    source_chunk_id:     str
    source_item:         str
    source_subsection:   Optional[str] = None
    reasoning:           Optional[str] = None

    model_config = {"use_enum_values": True}

    @field_validator("question")
    @classmethod
    def ensure_question_mark(cls, v: str) -> str:
        v = v.strip()
        if not v.endswith("?"):
            v += "?"
        return v

    @field_validator("source_passage")
    @classmethod
    def source_not_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 10:
            raise ValueError("source_passage must contain meaningful text")
        return v.strip()


class VerificationResult(BaseModel):
    qa_id:               str
    status:              VerificationStatus
    confidence:          str   = "medium"
    source_found:        bool  = False
    source_match_score:  float = 0.0
    numbers_verified:    bool  = True
    math_checked:        bool  = False
    math_passed:         Optional[bool] = None
    llm_verdict:         Optional[str]  = None
    llm_explanation:     Optional[str]  = None
    revision_suggestion: Optional[str]  = None

    model_config = {"use_enum_values": True}


class VerifiedQAPair(QAPair):
    verification:  VerificationResult
    verified_at:   datetime = Field(default_factory=datetime.utcnow)


class PipelineStats(BaseModel):
    total_chunks:          int   = 0
    total_generated:       int   = 0
    total_verified:        int   = 0
    total_rejected:        int   = 0
    verification_pass_rate: float = 0.0
    by_type:               Dict[str, int] = Field(default_factory=dict)
    by_difficulty:         Dict[str, int] = Field(default_factory=dict)
    by_section:            Dict[str, int] = Field(default_factory=dict)
    rejection_reasons:     Dict[str, int] = Field(default_factory=dict)


class FilingMetadata(BaseModel):
    company:          str
    cik:              str
    filing_type:      str = "10-K"
    fiscal_year_end:  str
    accession_number: str
    sec_url:          str


class PipelineOutput(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict) 
    filing:           FilingMetadata
    statistics:       PipelineStats
    qa_pairs:         List[VerifiedQAPair] = Field(default_factory=list)
    pipeline_version: str      = "1.0.0"
    generated_at:     datetime = Field(default_factory=datetime.utcnow)
    generator_model:  str      = ""
    verifier_model:   str      = ""

    def to_csv_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for qa in self.qa_pairs:
            rows.append({
                "id":                    qa.id,
                "question":              qa.question,
                "ground_truth_answer":   qa.ground_truth_answer,
                "source_passage":        qa.source_passage,
                "question_type":         qa.question_type,
                "difficulty":            qa.difficulty,
                "source_item":           qa.source_item,
                "source_subsection":     qa.source_subsection or "",
                "verification_status":   qa.verification.status,
                "verification_confidence": qa.verification.confidence,
                "source_match_score":    qa.verification.source_match_score,
                "math_checked":          qa.verification.math_checked,
                "math_passed":           qa.verification.math_passed,
            })
        return rows