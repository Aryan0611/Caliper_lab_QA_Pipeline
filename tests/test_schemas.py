"""Unit tests for Pydantic schemas"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import (
    QuestionType, DifficultyLevel, QAPair,
    Chunk, ContentType, VerificationResult, VerificationStatus
)


class TestQuestionType:
    def test_all_types_exist(self):
        assert QuestionType.FACT_EXTRACTION      == "fact_extraction"
        assert QuestionType.NUMERIC_CALCULATION  == "numeric_calculation"
        assert QuestionType.COMPARISON           == "comparison"
        assert QuestionType.MULTI_STEP_REASONING == "multi_step_reasoning"


class TestDifficultyLevel:
    def test_all_levels_exist(self):
        assert DifficultyLevel.EASY   == "easy"
        assert DifficultyLevel.MEDIUM == "medium"
        assert DifficultyLevel.HARD   == "hard"


class TestChunk:
    def test_valid_chunk(self):
        chunk = Chunk(
            chunk_id="msft_2025_item7_001",
            content="Revenue increased by 13% year over year to $318 billion.",
            item="Item 7",
            item_title="Management Discussion and Analysis",
            content_type=ContentType.TEXT_PARAGRAPH,
            contains_numbers=True,
            fiscal_years_mentioned=["2025", "2024"]
        )
        assert chunk.chunk_id == "msft_2025_item7_001"
        assert chunk.contains_numbers is True
        assert len(chunk.fiscal_years_mentioned) == 2

    def test_chunk_defaults(self):
        chunk = Chunk(
            chunk_id="test_001",
            content="Some content here",
            item="Item 1",
            item_title="Business"
        )
        assert chunk.content_type == ContentType.TEXT_PARAGRAPH
        assert chunk.token_count == 0
        assert chunk.contains_numbers is False


class TestQAPair:
    def test_valid_qa_pair(self):
        qa = QAPair(
            id="MSFT_2025_001",
            question="What was Microsoft's total revenue for FY2025?",
            ground_truth_answer="Microsoft's total revenue for FY2025 was $318,273,000,000.",
            source_passage="Total revenue for fiscal year 2025 was $318,273 million, compared to $281,685 million in 2024.",
            question_type=QuestionType.FACT_EXTRACTION,
            difficulty=DifficultyLevel.EASY,
            source_chunk_id="chunk_001",
            source_item="Item 8",
        )
        assert qa.id == "MSFT_2025_001"
        assert qa.question.endswith("?")

    def test_question_auto_adds_question_mark(self):
        qa = QAPair(
            id="TEST_002",
            question="What is the total revenue",
            ground_truth_answer="The total revenue is $318 billion.",
            source_passage="Total revenue for fiscal year 2025 was $318,273 million.",
            question_type=QuestionType.FACT_EXTRACTION,
            difficulty=DifficultyLevel.EASY,
            source_chunk_id="chunk_001",
            source_item="Item 8"
        )
        assert qa.question == "What is the total revenue?"

    def test_empty_source_passage_fails(self):
        with pytest.raises(Exception):
            QAPair(
                id="TEST_003",
                question="Question?",
                ground_truth_answer="Answer",
                source_passage="",
                question_type=QuestionType.FACT_EXTRACTION,
                difficulty=DifficultyLevel.EASY,
                source_chunk_id="chunk_001",
                source_item="Item 8"
            )


class TestVerificationResult:
    def test_valid_pass(self):
        result = VerificationResult(
            qa_id="MSFT_2025_001",
            status=VerificationStatus.PASS,
            confidence="high",
            source_found=True,
            source_match_score=0.95,
        )
        assert result.status == "pass"
        assert result.source_match_score == 0.95

    def test_valid_fail(self):
        result = VerificationResult(
            qa_id="MSFT_2025_002",
            status=VerificationStatus.FAIL,
            confidence="high",
            source_found=False,
            source_match_score=0.3,
        )
        assert result.status == "fail"
        assert result.source_found is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])