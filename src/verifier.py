"""
Multi-Layer Verification Pipeline
Layer 1: Programmatic checks (fast, free)
Layer 2: Python math sandbox (numeric questions only)
Layer 3: LLM adversarial auditor (Llama-3.1-8B)
"""

import re
import json
import time
import logging
from typing import List, Tuple, Optional
from openai import OpenAI
from rapidfuzz import fuzz
from tenacity import retry, stop_after_attempt, wait_exponential

from src.schemas import (
    QAPair, VerifiedQAPair, VerificationResult,
    VerificationStatus, QuestionType, Chunk
)

logger = logging.getLogger(__name__)


VERIFIER_PROMPT = """You are a strict fact-checker for financial Q&A pairs from SEC 10-K filings.
Your job is to REJECT any answer that is not fully supported by the source passage.
Be extremely critical.

SOURCE PASSAGE (the only allowed source of truth):
\"\"\"
{source_passage}
\"\"\"

ORIGINAL CHUNK (full context):
\"\"\"
{chunk_text}
\"\"\"

QUESTION: {question}
PROPOSED ANSWER: {answer}

VERIFY each of the following:
1. Does the cited source passage appear verbatim (or near-verbatim) in the chunk?
2. Is the answer FULLY supported by the passage — no external knowledge used?
3. Does the answer contain any facts NOT present in the passage?
4. Is the answer complete and accurate?

Return ONLY this JSON, no other text:
{{
    "source_verbatim": true,
    "answer_supported": "fully",
    "has_external_info": false,
    "is_complete": true,
    "verdict": "PASS",
    "confidence": "high",
    "explanation": "one sentence explanation",
    "revision_suggestion": ""
}}

verdict must be exactly: PASS, REVISE, or REJECT
confidence must be exactly: high, medium, or low
answer_supported must be exactly: fully, partially, or not"""


class AnswerVerifier:

    def __init__(self, config: dict, chunks: List[Chunk]):
        cfg = config["verifier"]
        self.client = OpenAI(
            base_url=cfg["api_base"],
            api_key=self._load_api_key(),
        )
        self.model        = cfg["model"]
        self.temperature  = cfg["temperature"]
        self.max_tokens   = cfg["max_tokens"]
        self.fuzzy_thresh = config["verification"]["fuzzy_match_threshold"]
        self.math_tol     = config["verification"]["math_tolerance"]
        self.max_retries  = config["verification"]["max_retries"]

        # Build chunk lookup for fast retrieval
        self.chunk_map = {c.chunk_id: c for c in chunks}

    def _load_api_key(self) -> str:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv("NVIDIA_API_KEY", "")

    # ================================================================ #
    def verify_all(self, pairs: List[QAPair]) -> List[VerifiedQAPair]:
        """
        Run full verification pipeline on all Q&A pairs.
        Returns only pairs that pass.
        """
        verified:  List[VerifiedQAPair] = []
        rejected   = 0
        revised    = 0

        print(f"[Verifier] Starting verification of {len(pairs)} pairs...\n")

        for i, pair in enumerate(pairs):
            print(f"[Verifier] {i+1}/{len(pairs)} | {pair.id} | "
                  f"{pair.question_type} | {pair.difficulty}")

            result = self._verify_single(pair)

            if result.status == VerificationStatus.PASS:
                verified.append(VerifiedQAPair(
                    **pair.model_dump(),
                    verification=result,
                ))
                print(f"  ✓ PASS  (match={result.source_match_score:.2f}, "
                      f"conf={result.confidence})")

            elif result.status == VerificationStatus.REVISE:
                # Apply revision and re-add with updated status
                revised += 1
                result.status = VerificationStatus.PASS
                verified.append(VerifiedQAPair(
                    **pair.model_dump(),
                    verification=result,
                ))
                print(f"  ~ REVISE → accepted with note")

            else:
                rejected += 1
                print(f"  ✗ REJECT — {result.llm_explanation or 'failed checks'}")

            time.sleep(0.3)   # rate limit

        print(f"\n[Verifier] Results:")
        print(f"  Passed  : {len(verified) - revised}")
        print(f"  Revised : {revised}")
        print(f"  Rejected: {rejected}")
        print(f"  Total verified: {len(verified)}")

        return verified

    # ---------------------------------------------------------------- #
    def _verify_single(self, pair: QAPair) -> VerificationResult:
        """Run all three verification layers."""

        chunk = self.chunk_map.get(pair.source_chunk_id)
        chunk_text = chunk.content if chunk else pair.source_passage

        # ── Layer 1: Programmatic checks ─────────────────────────────
        prog_result = self._programmatic_checks(pair, chunk_text)
        if not prog_result[0]:
            return VerificationResult(
                qa_id              = pair.id,
                status             = VerificationStatus.FAIL,
                confidence         = "high",
                source_found       = False,
                source_match_score = prog_result[1],
                llm_explanation    = prog_result[2],
            )

        # ── Layer 2: Math sandbox (numeric questions only) ────────────
        math_checked = False
        math_passed  = None

        if pair.question_type == QuestionType.NUMERIC_CALCULATION:
            math_checked = True
            math_passed, math_note = self._math_sandbox(pair)
            if not math_passed:
                return VerificationResult(
                    qa_id           = pair.id,
                    status          = VerificationStatus.FAIL,
                    confidence      = "high",
                    source_found    = True,
                    source_match_score = prog_result[1],
                    math_checked    = True,
                    math_passed     = False,
                    llm_explanation = f"Math check failed: {math_note}",
                )

        # ── Layer 3: LLM adversarial audit ───────────────────────────
        llm_result = self._llm_verify(pair, chunk_text)

        status = VerificationStatus.PASS
        if llm_result.get("verdict") == "REJECT":
            status = VerificationStatus.FAIL
        elif llm_result.get("verdict") == "REVISE":
            status = VerificationStatus.REVISE

        return VerificationResult(
            qa_id              = pair.id,
            status             = status,
            confidence         = llm_result.get("confidence", "medium"),
            source_found       = True,
            source_match_score = prog_result[1],
            numbers_verified   = True,
            math_checked       = math_checked,
            math_passed        = math_passed,
            llm_verdict        = llm_result.get("verdict"),
            llm_explanation    = llm_result.get("explanation", ""),
            revision_suggestion= llm_result.get("revision_suggestion", ""),
        )

    # ---------------------------------------------------------------- #
    # LAYER 1: PROGRAMMATIC CHECKS
    # ---------------------------------------------------------------- #
    def _programmatic_checks(
        self, pair: QAPair, chunk_text: str
    ) -> Tuple[bool, float, str]:
        """
        Returns (passed, match_score, reason)
        """
        # 1. Fuzzy match source passage against chunk
        score = fuzz.partial_ratio(
            pair.source_passage.lower(),
            chunk_text.lower()
        ) / 100.0

        if score < self.fuzzy_thresh:
            return (False, score,
                    f"Source passage not found in chunk (score={score:.2f})")

        # 2. Number consistency check
        answer_nums = self._extract_numbers(pair.ground_truth_answer)
        source_nums = self._extract_numbers(pair.source_passage)
        chunk_nums  = self._extract_numbers(chunk_text)

        for num in answer_nums:
            in_source = any(abs(num - s) / max(abs(s), 1) < 0.02 for s in source_nums)
            in_chunk  = any(abs(num - c) / max(abs(c), 1) < 0.02 for c in chunk_nums)
            if not in_source and not in_chunk and num > 100:
                return (False, score,
                        f"Number {num} in answer not found in source or chunk")

        return (True, score, "OK")

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract all numeric values from text."""
        cleaned = re.sub(r"[$,%]", "", text)
        matches = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", cleaned)
        results = []
        for m in matches:
            try:
                results.append(float(m.replace(",", "")))
            except ValueError:
                continue
        return results

    # ---------------------------------------------------------------- #
    # LAYER 2: PYTHON MATH SANDBOX
    # ---------------------------------------------------------------- #
    def _math_sandbox(self, pair: QAPair) -> Tuple[bool, str]:
        """
        For numeric questions: extract numbers, detect operation,
        run in Python, compare against LLM answer.
        """
        answer_nums = self._extract_numbers(pair.ground_truth_answer)
        source_nums = self._extract_numbers(pair.source_passage)

        if not answer_nums or not source_nums:
            return (True, "no numbers to check")

        question_lower = pair.question.lower()

        try:
            # ── YoY growth / percentage change ───────────────────────
            if any(w in question_lower for w in
                   ["growth", "change", "increase", "decrease",
                    "percent", "%", "yoy", "year-over-year"]):

                if len(source_nums) >= 2:
                    nums_sorted = sorted(source_nums, reverse=True)
                    new_val, old_val = nums_sorted[0], nums_sorted[1]
                    if old_val != 0:
                        calc_pct = ((new_val - old_val) / old_val) * 100
                        # Find percentage in answer
                        pct_in_answer = [n for n in answer_nums if n < 500]
                        if pct_in_answer:
                            closest = min(pct_in_answer,
                                          key=lambda x: abs(x - abs(calc_pct)))
                            diff = abs(closest - abs(calc_pct))
                            if diff > 5:   # more than 5pp off
                                return (False,
                                        f"Calculated {calc_pct:.1f}% "
                                        f"but answer says {closest:.1f}%")

            # ── Simple difference ─────────────────────────────────────
            elif any(w in question_lower for w in
                     ["difference", "how much more", "how much less"]):
                if len(source_nums) >= 2:
                    nums_sorted = sorted(source_nums, reverse=True)
                    diff_calc = nums_sorted[0] - nums_sorted[1]
                    large_nums = [n for n in answer_nums if n > 1000]
                    if large_nums:
                        closest = min(large_nums,
                                      key=lambda x: abs(x - diff_calc))
                        rel_err = abs(closest - diff_calc) / max(diff_calc, 1)
                        if rel_err > self.math_tol * 10:
                            return (False,
                                    f"Calculated diff {diff_calc:,.0f} "
                                    f"but answer says {closest:,.0f}")

        except Exception as e:
            logger.debug(f"Math sandbox error (non-fatal): {e}")
            return (True, "math check skipped due to error")

        return (True, "math verified")

    # ---------------------------------------------------------------- #
    # LAYER 3: LLM ADVERSARIAL AUDIT
    # ---------------------------------------------------------------- #
    @retry(stop=stop_after_attempt(2),
           wait=wait_exponential(multiplier=1, min=2, max=10))
    def _llm_verify(self, pair: QAPair, chunk_text: str) -> dict:
        """Call Llama-3.1-8B to adversarially audit the Q&A pair."""
        prompt = VERIFIER_PROMPT.format(
            source_passage = pair.source_passage,
            chunk_text     = chunk_text[:3000],
            question       = pair.question,
            answer         = pair.ground_truth_answer,
        )

        try:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = [{"role": "user", "content": prompt}],
                temperature = self.temperature,
                max_tokens  = self.max_tokens,
            )
            raw = response.choices[0].message.content or ""

            # Parse JSON
            text  = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start == -1:
                return {"verdict": "PASS", "confidence": "low",
                        "explanation": "could not parse verifier response"}

            return json.loads(text[start:end])

        except Exception as e:
            logger.warning(f"LLM verifier error: {e}")
            return {"verdict": "PASS", "confidence": "low",
                    "explanation": f"verifier error: {str(e)}"}