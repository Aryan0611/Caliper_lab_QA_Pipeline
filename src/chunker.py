"""
Document chunker for SEC 10-K sections.

The goal here is to keep related text together and avoid breaking tables
in the middle, since those chunks are later sent to the Q&A generator.
"""

import logging
import re
from typing import List

from src.schemas import Chunk, ContentType

logger = logging.getLogger(__name__)


# Token estimate
def _approx_tokens(text: str) -> int:
    return int(len(text) / 3.8)


class DocumentChunker:
    # Target sections
    TARGET_ITEMS = {
        "item 1",
        "item 1a",
        "item 1c",
        "item 7",
        "item 8",
    }

    def __init__(self, config: dict):
        # Chunk size settings
        cfg = config.get("chunking", {})
        self.min_chars = cfg.get("min_chunk_size", 300)
        self.max_chars = cfg.get("max_chunk_size", 6000)
        self.overlap = cfg.get("overlap", 150)
        self._chunk_counter = 0

    # Main chunking
    def chunk_sections(self, sections) -> List[Chunk]:
        """Chunk the sections that are useful for Q&A generation."""
        all_chunks: List[Chunk] = []

        for section in sections:
            item_key = section.item.lower().strip()

            if item_key not in self.TARGET_ITEMS:
                continue

            print(
                f"[Chunker] Chunking {section.item} - "
                f"{section.item_title} ({section.char_count:,} chars)"
            )

            chunks = self._chunk_section(section)
            all_chunks.extend(chunks)
            print(f"          -> {len(chunks)} chunks produced")

        print(f"\n[Chunker] Total chunks: {len(all_chunks)}")
        return all_chunks

    # Section splitting
    def _chunk_section(self, section) -> List[Chunk]:
        """Split one SEC section without breaking tables."""
        content = section.content
        chunks: List[Chunk] = []

        # Header-based splitting
        sub_sections = self._split_on_headers(content)

        for sub_title, sub_content in sub_sections:
            if len(sub_content.strip()) < self.min_chars:
                continue

            # Table-safe splitting
            has_table = "|" in sub_content and "---" in sub_content

            if has_table:
                blocks = self._split_preserve_tables(sub_content)
            else:
                blocks = [sub_content]

            for block in blocks:
                if len(block.strip()) < self.min_chars:
                    continue

                # Paragraph splitting for large blocks
                if len(block) > self.max_chars:
                    para_chunks = self._split_by_paragraphs(block)
                    for pc in para_chunks:
                        if len(pc.strip()) >= self.min_chars:
                            chunks.append(
                                self._make_chunk(
                                    content=pc,
                                    section=section,
                                    subsection=sub_title,
                                )
                            )
                else:
                    chunks.append(
                        self._make_chunk(
                            content=block,
                            section=section,
                            subsection=sub_title,
                        )
                    )

        return chunks

    # Header detection
    def _split_on_headers(self, text: str):
        """Split text when a line looks like a subsection heading."""
        lines = text.split("\n")
        sections = []
        cur_title = "General"
        cur_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if self._is_subheading(stripped):
                if cur_lines:
                    sections.append((cur_title, "\n".join(cur_lines)))
                cur_title = stripped
                cur_lines = []
            else:
                cur_lines.append(line)

        if cur_lines:
            sections.append((cur_title, "\n".join(cur_lines)))

        return sections if sections else [("General", text)]

    def _is_subheading(self, line: str) -> bool:
        """Heuristic for detecting subsection headings."""
        if not line:
            return False
        if line.startswith("|"):
            return False
        if len(line) > 120:
            return False
        if line.endswith("."):
            return False
        if line.endswith(","):
            return False
        if not line[0].isupper():
            return False
        if len(line) < 3:
            return False
        return True

    # Table handling
    def _split_preserve_tables(self, text: str) -> List[str]:
        """Split text into blocks while keeping Markdown tables intact."""
        blocks: List[str] = []
        cur_block: List[str] = []
        in_table = False

        for line in text.split("\n"):
            is_table_line = line.strip().startswith("|")

            if is_table_line and not in_table:
                if cur_block:
                    blocks.append("\n".join(cur_block))
                    cur_block = []
                in_table = True
                cur_block.append(line)

            elif not is_table_line and in_table:
                blocks.append("\n".join(cur_block))
                cur_block = []
                in_table = False
                cur_block.append(line)

            else:
                cur_block.append(line)

        if cur_block:
            blocks.append("\n".join(cur_block))

        return [b for b in blocks if b.strip()]

    # Paragraph splitting
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        Split oversized text at paragraph boundaries and merge small
        paragraphs up to max_chars.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if current_len + len(para) > self.max_chars and current:
                chunk_text = "\n\n".join(current)
                chunks.append(chunk_text)

                overlap_text = (
                    current[-1] if len(current[-1]) <= self.overlap * 4 else ""
                )
                current = [overlap_text, para] if overlap_text else [para]
                current_len = sum(len(c) for c in current)
            else:
                current.append(para)
                current_len += len(para)

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    # Chunk metadata
    def _make_chunk(self, content: str, section, subsection: str) -> Chunk:
        self._chunk_counter += 1

        content = content.strip()
        has_table = "|" in content and "---" in content
        has_numbers = bool(re.search(r"\$[\d,]+|\d+[%]|\d{4}", content))
        fiscal_years = re.findall(r"\b(20\d{2})\b", content)
        fiscal_years = sorted(set(fiscal_years), reverse=True)[:4]

        table_lines = sum(1 for line in content.split("\n") if line.startswith("|"))
        total_lines = max(len(content.split("\n")), 1)
        table_ratio = table_lines / total_lines

        if table_ratio > 0.55:
            ctype = ContentType.FINANCIAL_TABLE
        elif has_table:
            ctype = ContentType.MIXED
        else:
            ctype = ContentType.TEXT_PARAGRAPH

        item_slug = section.item.lower().replace(" ", "_")
        chunk_id = f"msft_2025_{item_slug}_{self._chunk_counter:04d}"

        return Chunk(
            chunk_id=chunk_id,
            content=content,
            item=section.item,
            item_title=section.item_title,
            subsection=subsection if subsection != "General" else None,
            content_type=ctype,
            token_count=_approx_tokens(content),
            char_count=len(content),
            contains_numbers=has_numbers,
            fiscal_years_mentioned=fiscal_years,
        )
