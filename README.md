# 10-K Q&A Pipeline

Automated pipeline that generates verified question-answer pairs from SEC 10-K filings for AI model benchmarking.

Built for Caliper Lab's assessment.
---
## Index

- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Pipeline Overview](#pipeline-overview)
- [Target Filing](#target-filing)
- [Q&A Distribution](#qa-distribution)
- [Output Format](#output-format)
- [Design Choices](#design-choices)
- [Known Limitations](#known-limitations)
- [Scaling to 1000+ Pairs](#scaling-to-multiple-documents-or-1000-pairs)
- [Running Tests](#running-tests)
- [Tech Stack](#tech-stack)
---

## Results

| Metric | Value |
|--------|-------|
| Source Document | Microsoft FY2025 10-K |
| Chunks Processed | 251 |
| Q&A Pairs Generated | 120 |
| Q&A Pairs Verified | **111** |
| Verification Pass Rate | **92.5%** |
| Total Runtime | ~67 minutes |

---
## Project Structure

```
10k-qa-pipeline/
├── main.py              # Entry point — run this
├── config.yaml          # All configuration parameters
├── requirements.txt
├── .env                 # Add your NVIDIA_API_KEY here
├── src/
│   ├── schemas.py       # Pydantic data models
│   ├── fetcher.py       # SEC EDGAR downloader
│   ├── parser.py        # HTML parser + table extractor
│   ├── chunker.py       # Hierarchical document chunker
│   ├── generator.py     # Q&A generation (Llama-3.1-70B)
│   ├── verifier.py      # Three-layer verification (Llama-3.1-8B)
│   └── pipeline.py      # Orchestrator connecting all stages
├── output/
│   ├── qa_pairs.json    # Full dataset — 111 verified pairs
│   ├── qa_pairs.csv     # CSV version
│   └── pipeline_log.json
├── data/raw/            # Downloaded 10-K cached here
└── tests/
    └── test_schemas.py
```

## How to Run

### Prerequisites

- Python 3.10+
- NVIDIA NIM API key — free at [build.nvidia.com](https://build.nvidia.com)

### Setup

```bash
git clone https://github.com/yourusername/10k-qa-pipeline.git
cd 10k-qa-pipeline

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file in the root:

```
NVIDIA_API_KEY=your_key_here
```

### Run

```bash
# Full pipeline — generates 120 Q&A pairs
python main.py

# Custom target
python main.py --target 150

# Debug mode
python main.py --debug
```

### Output

```
output/
├── qa_pairs.json       # Full dataset with metadata
├── qa_pairs.csv        # Flat CSV for analysis
└── pipeline_log.json   # Run statistics and timing
```

---

## Pipeline Overview

```
SEC EDGAR
    │
    ▼
Fetch HTML (fetcher.py)
    │  Downloads msft-20250630.htm, caches locally
    ▼
Parse + Clean (parser.py)
    │  Strips HTML noise, converts tables → Markdown,
    │  detects SEC Item boundaries (Item 1, 1A, 7, 8…)
    ▼
Hierarchical Chunk (chunker.py)
    │  Level 1: SEC Item boundaries
    │  Level 2: Subsection headers
    │  Level 3: Paragraph breaks (if still too large)
    │  Tables are never split mid-table
    │  Each chunk tagged: text_paragraph / financial_table / mixed
    ▼
Generate Q&A (generator.py)  ←  Llama-3.1-70B
    │  Routes chunk type to matching question type:
    │    financial_table  →  numeric_calculation
    │    text_paragraph   →  fact_extraction / comparison / multi_step_reasoning
    │  One question per chunk, deduplication enforced
    ▼
Verify (verifier.py)  ←  Llama-3.1-8B
    │  Layer 1 — Programmatic: fuzzy match source passage against chunk (≥0.80)
    │             number consistency check across answer and source
    │  Layer 2 — Math sandbox: for numeric questions, extract numbers,
    │             run calculation in Python, compare against LLM answer (±0.5%)
    │  Layer 3 — LLM audit: Llama-3.1-8B at temperature 0.0,
    │             prompted to adversarially reject unsupported answers
    ▼
Output JSON + CSV
```

---

## Target Filing

| Field | Value |
|-------|-------|
| Company | Microsoft Corporation |
| CIK | 0000789019 |
| Filing Type | Form 10-K |
| Fiscal Year End | June 30, 2025 |
| Accession No. | 0000950170-25-100235 |

---

## Q&A Distribution

**By question type:**

| Type | Count | % |
|------|-------|---|
| fact_extraction | 51 | 46% |
| comparison | 36 | 32% |
| multi_step_reasoning | 21 | 19% |
| numeric_calculation | 12 | 11% |

**By difficulty:**

| Difficulty | Count | % |
|------------|-------|---|
| medium | 51 | 46% |
| hard | 38 | 34% |
| easy | 31 | 28% |

**By source section:**

| Section | Description | Chunks |
|---------|-------------|--------|
| Item 1 | Business | 44 |
| Item 1A | Risk Factors | 30 |
| Item 1C | Cybersecurity | 4 |
| Item 7 | MD&A | 50 |
| Item 8 | Financial Statements | 123 |

---

## Output Format

Each row in the dataset contains:

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (MSFT_2025_XXXX) |
| `question` | The generated question |
| `ground_truth_answer` | Correct answer derived from the filing |
| `source_passage` | Exact verbatim text from the 10-K supporting the answer |
| `question_type` | fact_extraction / numeric_calculation / comparison / multi_step_reasoning |
| `difficulty` | easy / medium / hard |
| `source_item` | SEC Item number (Item 1, Item 7, etc.) |
| `source_subsection` | Subsection within the Item |
| `verification_status` | pass / revise |
| `source_match_score` | Fuzzy match score 0–1 |
| `math_checked` | Whether Python math sandbox ran |
| `math_passed` | Whether math check passed (null if non-numeric) |

### Sample Q&A Pairs

**Fact Extraction — Easy**
```json
{
  "question": "What are the three operating segments of Microsoft Corporation?",
  "ground_truth_answer": "Microsoft's three operating segments are: (1) Productivity and Business Processes, (2) Intelligent Cloud, and (3) More Personal Computing.",
  "source_passage": "We operate our business and report our financial performance using three segments: Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.",
  "question_type": "fact_extraction",
  "difficulty": "easy",
  "source_item": "Item 1"
}
```

**Numeric Calculation — Medium**
```json
{
  "question": "What was Microsoft's year-over-year revenue growth rate for fiscal year 2025?",
  "ground_truth_answer": "Microsoft's revenue grew approximately 13% year-over-year, from $281,685 million in FY2024 to $318,273 million in FY2025, an increase of $36,588 million.",
  "source_passage": "Total revenue for fiscal year 2025 was $318,273 million, compared to $281,685 million for fiscal year 2024, an increase of $36,588 million or 13%.",
  "question_type": "numeric_calculation",
  "difficulty": "medium",
  "source_item": "Item 8"
}
```

**Multi-Step Reasoning — Hard**
```json
{
  "question": "According to the risk factors, how could Microsoft's capital investments in AI infrastructure impact operating margins if customer demand does not grow proportionally?",
  "ground_truth_answer": "If customer demand for AI services does not grow proportionally with Microsoft's infrastructure investments, the company faces margin compression from elevated depreciation on underutilized data center capacity — long-term capital allocations that cannot be quickly adjusted.",
  "source_passage": "Our deployment of AI infrastructure requires long-term capital allocations. If customer demand doesn't grow proportionally, our operating metrics could be materially impacted by unused capacity costs and accelerated depreciation.",
  "question_type": "multi_step_reasoning",
  "difficulty": "hard",
  "source_item": "Item 1A"
}
```

---

## Design Choices

### Two different model sizes for generation vs. verification

Generator uses Llama-3.1-70B; verifier uses Llama-3.1-8B. Using the same model for both creates self-confirmation bias — the model validates its own hallucinations. A different model at temperature 0.0 provides genuinely independent judgment.

### Hierarchical chunking over fixed-size splitting

Fixed-size chunking (e.g. 1000 tokens) breaks financial tables mid-row and severs sentences that belong together. Section-aware splitting respects the regulatory structure of 10-K filings. Tables are never split — they are kept as intact Markdown blocks.

### Three verification layers, cheap first

Programmatic checks (fuzzy match, number consistency) run first and reject failures immediately at no API cost. Python math sandbox catches calculation errors deterministically. LLM audit only runs if layers 1 and 2 pass, keeping costs low.

### Element tagging for prompt routing

Each chunk is tagged as `text_paragraph`, `financial_table`, or `mixed` during chunking. This tag routes the chunk to the appropriate prompt template — table chunks get numeric/calculation prompts; narrative chunks get reasoning/comparison prompts. Without this, you get weak math questions from prose and poor reasoning questions from tables.

### No vector database or embeddings

For single-document generation, vector search adds complexity without benefit. The pipeline iterates every chunk sequentially — there is no retrieval step. ChromaDB and embeddings are noted in the scaling section for multi-document scenarios where cross-document search becomes necessary.

### Pydantic v2 schema enforcement

All LLM output is validated through Pydantic models before entering the pipeline. Invalid JSON, missing fields, empty source passages, and type errors are caught and discarded automatically without crashing the pipeline.

---

## Known Limitations

**1. Complex nested tables lose structure**
Some deeply nested HTML tables in the filing lose column alignment during HTML→Markdown conversion. Numeric questions sourced from these may reference malformed data. Mitigation: visual diff check on table-sourced Q&A pairs.

**2. Number normalisation ambiguity**
`$318.3B`, `$318,273M`, and `$318,273,000,000` represent the same value but appear differently in the document. The math sandbox may flag valid answers as mismatches. Mitigation: normalise all numbers to raw integers at parse time.

**3. No cross-section questions**
Questions are generated from individual chunks. The pipeline cannot produce questions that require connecting information from two distant sections — for example, linking a risk described in Item 1A to a specific financial figure in Item 8.

**4. Rate limits slow the pipeline**
NVIDIA NIM free tier allows approximately 30 requests per minute. The full pipeline takes ~67 minutes for 120 pairs. Retry logic with exponential backoff is implemented, but large runs will still be slow.

**5. Section detection is regex-based**
The parser detects `Item 1`, `Item 1A`, etc. using pattern matching. Different companies format these headings differently. The regex works reliably for Microsoft's filing but may miss boundaries in other filings without adjustment.

**6. Numeric question coverage is lower than target**
Only 12 numeric calculation questions were generated (target was ~25). Many text-heavy chunks lack structured numeric data. Financial statement chunks contain numbers but the Markdown table format sometimes confuses the generator prompt.

---

## Scaling to Multiple Documents or 1,000+ Pairs

### More pairs from one document

Increase `questions_per_chunk` from 1 to 3 in `config.yaml`. Process footnotes and notes-to-financials sections currently skipped. Add a rephrasing pass to generate question variants from existing verified pairs.

### Multiple documents — manifest-driven processing

Build a central manifest (SQLite or PostgreSQL) tracking `filing_id`, `ticker`, `cik`, `year`, `processing_status`. The pipeline checks the manifest on startup and processes any `pending` entries. To add 100 new companies, insert 100 rows — no code changes needed.

### Diversity at scale — K-Means topic clustering

At 1,000+ pairs from 50+ documents, some topics get over-sampled. Embed all chunks using a sentence encoder, run K-Means (K=20 clusters), sample an equal number of Q&A pairs from each cluster. This guarantees topical diversity across the full corpus rather than concentrating on the largest sections.

### Cost and speed at scale — async batch API

Replace synchronous requests with async batch API calls. Most providers process batch jobs at 50% cost discount and without standard rate limits. For 10,000 pairs across 100 documents, this reduces runtime from days to hours.

### Storage at scale — PostgreSQL + pgvector

Replace JSON file storage with PostgreSQL for structured querying and pgvector for semantic deduplication across documents. This also enables cross-document question generation by retrieving semantically similar chunks from different filings.

---
---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| LLM Generator | Llama-3.1-70B via NVIDIA NIM | — |
| LLM Verifier | Llama-3.1-8B via NVIDIA NIM | — |
| Schema validation | pydantic | 2.5+ |
| HTML parsing | beautifulsoup4 + lxml | 4.12+ |
| Fuzzy matching | rapidfuzz | 3.6+ |
| HTTP client | requests + openai | — |
| Output | pandas | 2.1+ |
