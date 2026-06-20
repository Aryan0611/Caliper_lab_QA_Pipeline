"""Fetch and cache the SEC 10-K filing used by the pipeline."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


@dataclass
class FilingDocument:
    company: str
    cik: str
    accession_number: str
    url: str
    raw_html: str = ""
    local_path: str = ""


class SECFetcher:
    def __init__(self, config: dict):
        # Config
        self.config = config
        self.user_agent = config["sec"]["user_agent"]
        self.delay = config["sec"]["rate_limit_delay"]
        self.filing_cfg = config["filing"]

        # SEC request session
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": "www.sec.gov",
            }
        )

    # Fetch filing
    def fetch(self, save_dir: str = "data/raw") -> FilingDocument:
        """
        Download the filing if needed. If it already exists locally,
        load it from disk to avoid repeated SEC requests.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        local_path = Path(save_dir) / "msft-20250630.htm"

        # Cache lookup
        if local_path.exists():
            logger.info("Loading cached filing from %s", local_path)
            print(f"[Fetcher] Using cached file: {local_path}")
            html = local_path.read_text(encoding="utf-8", errors="replace")

            return FilingDocument(
                company=self.filing_cfg["company"],
                cik=self.filing_cfg["cik"],
                accession_number=self.filing_cfg["accession_number"],
                url=self.filing_cfg["url"],
                raw_html=html,
                local_path=str(local_path),
            )

        # Download from SEC
        url = self.filing_cfg["url"]
        print("[Fetcher] Downloading from SEC EDGAR...")
        print(f"[Fetcher] URL: {url}")

        time.sleep(self.delay)

        response = self.session.get(url, timeout=120)
        response.raise_for_status()

        html = response.text
        local_path.write_text(html, encoding="utf-8")

        size_mb = local_path.stat().st_size / 1_048_576
        print(
            f"[Fetcher] Saved -> {local_path} "
            f"({len(html):,} characters, {size_mb:.1f} MB)"
        )

        return FilingDocument(
            company=self.filing_cfg["company"],
            cik=self.filing_cfg["cik"],
            accession_number=self.filing_cfg["accession_number"],
            url=url,
            raw_html=html,
            local_path=str(local_path),
        )
