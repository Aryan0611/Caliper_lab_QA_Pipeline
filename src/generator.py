"""Generate Q&A pairs from document chunks using NVIDIA NIM."""

import json
import logging
import re
import time
from typing import List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.schemas import Chunk, ContentType, DifficultyLevel, QAPair, QuestionType

logger = logging.getLogger(__name__)


# Prompt templates
PROMPTS = {
    QuestionType.FACT_EXTRACTION: """You are a financial analyst creating evaluation questions from SEC 10-K filings.

PASSAGE FROM {item} - {item_title}:
\"\"\"
{chunk_text}
\"\"\"

TASK: Generate {n} FACT EXTRACTION question(s) at {difficulty} difficulty.

Fact extraction questions ask about specific facts, names, entities, dates, or descriptions stated directly in the passage.

Easy: single fact, directly stated
Medium: requires identifying specific detail among many
Hard: requires combining two related facts from the passage

RULES:
- Question must be answerable ONLY from the passage above
- source_passage must be EXACT text copied from the passage (verbatim)
- Do not ask about the company name or obvious facts

Return ONLY a JSON array, no other text:
[
  {{
    "question": "...",
    "answer": "...",
    "source_passage": "exact verbatim quote from passage",
    "reasoning": "why this answer is correct"
  }}
]""",

    QuestionType.NUMERIC_CALCULATION: """You are a financial analyst creating evaluation questions from SEC 10-K filings.

PASSAGE FROM {item} - {item_title}:
\"\"\"
{chunk_text}
\"\"\"

TASK: Generate {n} NUMERIC CALCULATION question(s) at {difficulty} difficulty.

Numeric calculation questions require working with numbers: computing growth rates, differences, ratios, or percentages using figures from the passage.

Easy: single number lookup or simple difference
Medium: percentage change or ratio calculation
Hard: multi-step calculation combining several figures

RULES:
- All numbers used must come from the passage
- Show the calculation in the answer
- source_passage must be EXACT text copied from the passage containing the numbers
- Only generate this type if the passage actually contains numbers

Return ONLY a JSON array, no other text:
[
  {{
    "question": "...",
    "answer": "...",
    "source_passage": "exact verbatim quote from passage containing the numbers",
    "reasoning": "step-by-step calculation"
  }}
]""",

    QuestionType.COMPARISON: """You are a financial analyst creating evaluation questions from SEC 10-K filings.

PASSAGE FROM {item} - {item_title}:
\"\"\"
{chunk_text}
\"\"\"

TASK: Generate {n} COMPARISON question(s) at {difficulty} difficulty.

Comparison questions ask the reader to compare two or more items: segments, years, metrics, products, or risks.

Easy: direct comparison of two values
Medium: comparison with explanation of difference
Hard: comparison requiring synthesis of multiple data points

RULES:
- Both items being compared must exist in the passage
- source_passage must be EXACT text copied from the passage
- Be specific about what is being compared

Return ONLY a JSON array, no other text:
[
  {{
    "question": "...",
    "answer": "...",
    "source_passage": "exact verbatim quote from passage",
    "reasoning": "how the comparison was made"
  }}
]""",

    QuestionType.MULTI_STEP_REASONING: """You are a financial analyst creating evaluation questions from SEC 10-K filings.

PASSAGE FROM {item} - {item_title}:
\"\"\"
{chunk_text}
\"\"\"

TASK: Generate {n} MULTI-STEP REASONING question(s) at {difficulty} difficulty.

Multi-step reasoning questions require connecting multiple pieces of information: cause-effect relationships, implications of risks, or conclusions drawn from several facts.

Medium: two-step logical connection
Hard: three or more steps, causal chain reasoning

RULES:
- All reasoning steps must be grounded in the passage
- source_passage must be EXACT text copied from the passage
- Question should require genuine reasoning, not just lookup

Return ONLY a JSON array, no other text:
[
  {{
    "question": "...",
    "answer": "...",
    "source_passage": "exact verbatim quote from passage",
    "reasoning": "step-by-step logical chain"
  }}
]""",
}


class QAGenerator:
    # Question type routing by SEC section
    SECTION_ROUTING = {
        "item 1": [
            (QuestionType.FACT_EXTRACTION, DifficultyLevel.EASY),
            (QuestionType.FACT_EXTRACTION, DifficultyLevel.MEDIUM),
            (QuestionType.COMPARISON, DifficultyLevel.MEDIUM),
        ],
        "item 1a": [
            (QuestionType.FACT_EXTRACTION, DifficultyLevel.EASY),
            (QuestionType.MULTI_STEP_REASONING, DifficultyLevel.HARD),
            (QuestionType.COMPARISON, DifficultyLevel.MEDIUM),
        ],
        "item 1c": [
            (QuestionType.FACT_EXTRACTION, DifficultyLevel.MEDIUM),
            (QuestionType.MULTI_STEP_REASONING, DifficultyLevel.HARD),
        ],
        "item 7": [
            (QuestionType.NUMERIC_CALCULATION, DifficultyLevel.MEDIUM),
            (QuestionType.COMPARISON, DifficultyLevel.HARD),
            (QuestionType.MULTI_STEP_REASONING, DifficultyLevel.HARD),
        ],
        "item 8": [
            (QuestionType.NUMERIC_CALCULATION, DifficultyLevel.EASY),
            (QuestionType.NUMERIC_CALCULATION, DifficultyLevel.MEDIUM),
            (QuestionType.FACT_EXTRACTION, DifficultyLevel.EASY),
        ],
    }

    def __init__(self, config: dict):
        # Model config
        cfg = config["generator"]
        self.client = OpenAI(
            base_url=cfg["api_base"],
            api_key=self._load_api_key(),
        )
        self.model = cfg["model"]
        self.temperature = cfg["temperature"]
        self.max_tokens = cfg["max_tokens"]
        self._qa_counter = 0

    def _load_api_key(self) -> str:
        import os

        from dotenv import load_dotenv

        load_dotenv()
        key = os.getenv("NVIDIA_API_KEY", "")
        if not key:
            raise ValueError("NVIDIA_API_KEY not found in .env file")
        return key

    # Main generation loop
    def generate_from_chunks(
        self,
        chunks: List[Chunk],
        target: int = 120,
    ) -> List[QAPair]:
        """Generate Q&A pairs until the target count is reached."""
        all_pairs: List[QAPair] = []
        seen_questions = set()

        # Keep small chunks out of the model calls.
        eligible = [chunk for chunk in chunks if chunk.char_count >= 300]

        print(f"[Generator] {len(eligible)} eligible chunks")
        print(f"[Generator] Target: {target} Q&A pairs\n")

        for i, chunk in enumerate(eligible):
            if len(all_pairs) >= target:
                break

            item_key = chunk.item.lower().strip()
            routes = self.SECTION_ROUTING.get(
                item_key,
                [(QuestionType.FACT_EXTRACTION, DifficultyLevel.MEDIUM)],
            )

            # Table chunks usually produce better numeric questions.
            if chunk.content_type == ContentType.FINANCIAL_TABLE:
                routes = [
                    (QuestionType.NUMERIC_CALCULATION, DifficultyLevel.MEDIUM),
                    (QuestionType.NUMERIC_CALCULATION, DifficultyLevel.HARD),
                    (QuestionType.FACT_EXTRACTION, DifficultyLevel.EASY),
                ]

            route_index = i % len(routes)
            q_type, difficulty = routes[route_index]

            if q_type == QuestionType.NUMERIC_CALCULATION and not chunk.contains_numbers:
                q_type = QuestionType.FACT_EXTRACTION

            print(
                f"[Generator] Chunk {i + 1}/{len(eligible)} | "
                f"{chunk.item} | {q_type} | {difficulty} | "
                f"{chunk.char_count} chars"
            )

            pairs = self._generate_for_chunk(chunk, q_type, difficulty, n=1)

            # Deduplication
            for pair in pairs:
                q_lower = pair.question.lower().strip()
                if q_lower in seen_questions:
                    continue

                seen_questions.add(q_lower)
                all_pairs.append(pair)
                print(f"  ok [{len(all_pairs):>3}] {pair.question[:80]}...")

            time.sleep(0.5)

        print(f"\n[Generator] Generated {len(all_pairs)} unique Q&A pairs")
        return all_pairs

    # Model call
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
    )
    def _generate_for_chunk(
        self,
        chunk: Chunk,
        q_type: QuestionType,
        difficulty: DifficultyLevel,
        n: int = 1,
    ) -> List[QAPair]:
        """Call the model and parse the response into QAPair objects."""
        prompt = PROMPTS[q_type].format(
            item=chunk.item,
            item_title=chunk.item_title,
            chunk_text=chunk.content[:4000],
            difficulty=difficulty,
            n=n,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw_text = response.choices[0].message.content or ""
            return self._parse_response(raw_text, chunk, q_type, difficulty)

        except Exception as exc:
            logger.warning("Generation failed for chunk %s: %s", chunk.chunk_id, exc)
            return []

    # Response parsing
    def _parse_response(
        self,
        raw_text: str,
        chunk: Chunk,
        q_type: QuestionType,
        difficulty: DifficultyLevel,
    ) -> List[QAPair]:
        """Extract the JSON array returned by the model."""
        text = re.sub(r"```(?:json)?", "", raw_text).strip()
        text = text.strip("`").strip()

        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            logger.warning("No JSON array found in response: %s", text[:200])
            return []

        try:
            items = json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error: %s | text: %s", exc, text[start:end][:200])
            return []

        pairs = []
        for item in items:
            if not isinstance(item, dict):
                continue

            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            source = item.get("source_passage", "").strip()

            if not question or not answer or not source:
                continue
            if len(source) < 10:
                continue

            self._qa_counter += 1
            try:
                pair = QAPair(
                    id=f"MSFT_2025_{self._qa_counter:04d}",
                    question=question,
                    ground_truth_answer=answer,
                    source_passage=source,
                    question_type=q_type,
                    difficulty=difficulty,
                    source_chunk_id=chunk.chunk_id,
                    source_item=chunk.item,
                    source_subsection=chunk.subsection,
                    reasoning=item.get("reasoning", ""),
                )
                pairs.append(pair)
            except Exception as exc:
                logger.warning("QAPair validation failed: %s", exc)

        return pairs
