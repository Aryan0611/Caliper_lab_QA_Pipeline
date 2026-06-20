"""
HTML Parser for SEC 10-K Filings
Extracts clean text, converts tables to Markdown, tags content types,
and splits the document into named SEC sections.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


@dataclass
class Section:
    item:         str
    item_title:   str
    content:      str
    content_type: str = "text_paragraph"   # text_paragraph | financial_table | mixed
    has_tables:   bool = False
    char_count:   int  = 0


class SECParser:

    # ── Known SEC Item labels ────────────────────────────────────────
    ITEM_MAP = {
        "item 1":   "Business",
        "item 1a":  "Risk Factors",
        "item 1b":  "Unresolved Staff Comments",
        "item 1c":  "Cybersecurity",
        "item 2":   "Properties",
        "item 3":   "Legal Proceedings",
        "item 4":   "Mine Safety Disclosures",
        "item 5":   "Market for Registrant Equity",
        "item 6":   "Reserved",
        "item 7":   "Management Discussion and Analysis",
        "item 7a":  "Quantitative and Qualitative Disclosures",
        "item 8":   "Financial Statements",
        "item 9":   "Changes in and Disagreements with Accountants",
        "item 9a":  "Controls and Procedures",
        "item 9b":  "Other Information",
        "item 10":  "Directors and Executive Officers",
        "item 11":  "Executive Compensation",
        "item 12":  "Security Ownership",
        "item 13":  "Certain Relationships",
        "item 14":  "Principal Accountant Fees",
        "item 15":  "Exhibits",
    }

    # ── Regex to detect Item headings ────────────────────────────────
    ITEM_PATTERN = re.compile(
        r"^item\s+(1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9|10|11|12|13|14|15)\b",
        re.IGNORECASE
    )

    def __init__(self, config: dict):
        self.config = config

    # ================================================================ #
    def parse(self, html: str) -> List[Section]:
        """
        Full parse pipeline:
        1. Clean HTML
        2. Convert tables → Markdown
        3. Extract text
        4. Split by SEC Item
        5. Tag content type
        """
        print("[Parser] Starting HTML parse...")
        soup = self._clean_html(html)

        print("[Parser] Converting tables to Markdown...")
        self._replace_tables_with_markdown(soup)

        print("[Parser] Extracting text...")
        full_text = self._extract_text(soup)

        print("[Parser] Splitting into sections...")
        sections = self._split_into_sections(full_text)

        print(f"[Parser] Found {len(sections)} sections")
        for s in sections:
            print(f"         {s.item:10s} | {s.item_title:45s} | "
                  f"{s.char_count:>7,} chars | tables={s.has_tables}")

        return sections

    # ---------------------------------------------------------------- #
    # 1. CLEAN HTML
    # ---------------------------------------------------------------- #
    def _clean_html(self, html: str) -> BeautifulSoup:
        soup = BeautifulSoup(html, "lxml")

        # Remove non-content tags
        for tag in soup(["script", "style", "meta", "link",
                          "noscript", "header", "footer", "nav"]):
            tag.decompose()

        # Remove XBRL inline tags but keep their text
        for tag in soup.find_all(re.compile(r"^ix:", re.I)):
            tag.unwrap()

        return soup

    # ---------------------------------------------------------------- #
    # 2. TABLE → MARKDOWN
    # ---------------------------------------------------------------- #
    def _replace_tables_with_markdown(self, soup: BeautifulSoup) -> None:
        for table in soup.find_all("table"):
            md = self._table_to_markdown(table)
            if md:
                placeholder = soup.new_tag("p")
                placeholder.string = md
                table.replace_with(placeholder)
            else:
                table.decompose()

    def _table_to_markdown(self, table: Tag) -> Optional[str]:
        rows = []
        for tr in table.find_all("tr"):
            cells = []
            for cell in tr.find_all(["td", "th"]):
                text = cell.get_text(separator=" ", strip=True)
                text = re.sub(r"\s+", " ", text)
                # Normalise dollar amounts
                text = self._normalise_number(text)
                cells.append(text)
            if any(cells):
                rows.append(cells)

        if not rows:
            return None

        # Make all rows same width
        width = max(len(r) for r in rows)
        rows  = [r + [""] * (width - len(r)) for r in rows]

        # Build Markdown
        header    = "| " + " | ".join(rows[0]) + " |"
        separator = "| " + " | ".join(["---"] * width) + " |"
        body      = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])

        return "\n".join(filter(None, [header, separator, body]))

    # ---------------------------------------------------------------- #
    # 3. EXTRACT CLEAN TEXT
    # ---------------------------------------------------------------- #
    def _extract_text(self, soup: BeautifulSoup) -> str:
        text = soup.get_text(separator="\n", strip=True)
        # Collapse excessive blank lines
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        return text

    # ---------------------------------------------------------------- #
    # 4. SPLIT BY SEC ITEM
    # ---------------------------------------------------------------- #
    def _split_into_sections(self, text: str) -> List[Section]:
        lines   = text.split("\n")
        sections: List[Section] = []

        current_item:    Optional[str]       = None
        current_title:   str                 = "Preamble"
        current_lines:   List[str]           = []

        def _save(item, title, lines_buf):
            content = "\n".join(lines_buf).strip()
            if len(content) < 100:      # skip near-empty sections
                return
            has_tables   = "| --- |" in content or "|---|" in content
            content_type = self._tag_content_type(content, has_tables)
            sections.append(Section(
                item         = item or "preamble",
                item_title   = title,
                content      = content,
                content_type = content_type,
                has_tables   = has_tables,
                char_count   = len(content),
            ))

        for line in lines:
            stripped = line.strip()
            match    = self.ITEM_PATTERN.match(stripped)

            if match:
                # Save previous section
                if current_lines:
                    _save(current_item, current_title, current_lines)

                # Start new section
                key           = "item " + match.group(1).lower()
                current_item  = key.title()
                current_title = self.ITEM_MAP.get(key, "Unknown")
                current_lines = [line]
            else:
                current_lines.append(line)

        # Save last section
        if current_lines:
            _save(current_item, current_title, current_lines)

        return sections

    # ---------------------------------------------------------------- #
    # 5. TAG CONTENT TYPE
    # ---------------------------------------------------------------- #
    def _tag_content_type(self, content: str, has_tables: bool) -> str:
        table_lines = sum(1 for l in content.split("\n") if l.startswith("|"))
        total_lines = max(len(content.split("\n")), 1)
        table_ratio = table_lines / total_lines

        if table_ratio > 0.6:
            return "financial_table"
        elif has_tables:
            return "mixed"
        else:
            return "text_paragraph"

    # ---------------------------------------------------------------- #
    # HELPERS
    # ---------------------------------------------------------------- #
    def _normalise_number(self, text: str) -> str:
        """Convert shorthand like $318.3B → $318,300,000,000"""
        def replace_match(m):
            num    = float(m.group(1).replace(",", ""))
            suffix = m.group(2).upper()
            mult   = {"B": 1_000_000_000, "M": 1_000_000, "K": 1_000}.get(suffix, 1)
            return f"${int(num * mult):,}"

        return re.sub(
            r"\$\s*([\d,]+\.?\d*)\s*([BMK])\b",
            replace_match,
            text,
            flags=re.IGNORECASE
        )