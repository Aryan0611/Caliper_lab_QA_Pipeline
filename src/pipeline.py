"""
Main Pipeline Orchestrator
Coordinates: Fetch → Parse → Chunk → Generate → Verify → Output

Usage:
    from src.pipeline import QAPipeline
    pipeline = QAPipeline("config.yaml")
    results = pipeline.run()
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import yaml
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.schemas import (
    Chunk, QAPair, VerifiedQAPair,
    PipelineOutput, PipelineStats, FilingMetadata,
    QuestionType, DifficultyLevel, VerificationStatus
)
from src.fetcher import SECFetcher
from src.parser import SECParser
from src.chunker import DocumentChunker
from src.generator import QAGenerator
from src.verifier import AnswerVerifier

logger = logging.getLogger(__name__)
console = Console()


class QAPipeline:
    """
    End-to-end pipeline for generating verified Q&A pairs from SEC 10-K filings.
    """

    def __init__(self, config_path: str = "config.yaml", debug: bool = False):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            debug: Enable debug logging
        """
        self.config_path = config_path
        self.debug = debug
        self.config = self._load_config()
        
        # Setup logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        
        # Pipeline state
        self.sections = []
        self.chunks: List[Chunk] = []
        self.raw_pairs: List[QAPair] = []
        self.verified_pairs: List[VerifiedQAPair] = []
        self.stats = PipelineStats()
        
        # Timing
        self.start_time: Optional[float] = None
        self.stage_times: Dict[str, float] = {}

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            return yaml.safe_load(f)

    # ================================================================ #
    # MAIN ENTRY POINTS
    # ================================================================ #
    def run(self, target_pairs: int = 120) -> Dict[str, Any]:
        """
        Run the complete pipeline end-to-end.
        
        Args:
            target_pairs: Target number of Q&A pairs to generate
            
        Returns:
            Dictionary with results summary
        """
        self.start_time = time.time()
        
        console.print("\n[bold cyan]╔══════════════════════════════════════════════════════╗[/bold cyan]")
        console.print("[bold cyan]║       10-K Q&A PIPELINE - FULL RUN                   ║[/bold cyan]")
        console.print("[bold cyan]╚══════════════════════════════════════════════════════╝[/bold cyan]\n")
        
        try:
            # Stage 1: Fetch
            self._run_stage_fetch()
            
            # Stage 2: Parse
            self._run_stage_parse()
            
            # Stage 3: Chunk
            self._run_stage_chunk()
            
            # Stage 4: Generate
            self._run_stage_generate(target=target_pairs)
            
            # Stage 5: Verify
            self._run_stage_verify()
            
            # Stage 6: Output
            output_files = self._run_stage_output()
            
            # Final summary
            self._print_final_summary()
            
            return {
                "success": True,
                "total_generated": len(self.raw_pairs),
                "total_verified": len(self.verified_pairs),
                "pass_rate": len(self.verified_pairs) / max(len(self.raw_pairs), 1),
                "output_files": output_files,
                "duration_seconds": time.time() - self.start_time,
                "statistics": self.stats.model_dump(),
            }
            
        except Exception as e:
            console.print(f"\n[bold red]Pipeline failed: {e}[/bold red]")
            if self.debug:
                import traceback
                traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
            }

    def run_stage(self, stage: str) -> Dict[str, Any]:
        """Run a specific pipeline stage."""
        stage_map = {
            "fetch": self._run_stage_fetch,
            "parse": self._run_stage_parse,
            "chunk": self._run_stage_chunk,
            "generate": self._run_stage_generate,
            "verify": self._run_stage_verify,
            "output": self._run_stage_output,
        }
        
        if stage not in stage_map:
            raise ValueError(f"Unknown stage: {stage}. Valid: {list(stage_map.keys())}")
        
        return stage_map[stage]()

    # ================================================================ #
    # PIPELINE STAGES
    # ================================================================ #
    def _run_stage_fetch(self) -> Dict[str, Any]:
        """Stage 1: Fetch document from SEC EDGAR."""
        console.print("[bold yellow]▶ STAGE 1: Fetching document...[/bold yellow]")
        start = time.time()
        
        fetcher = SECFetcher(self.config)
        self.document = fetcher.fetch()
        
        self.stage_times["fetch"] = time.time() - start
        console.print(f"  [green]✓ Fetched {len(self.document.raw_html):,} bytes "
                     f"({self.stage_times['fetch']:.1f}s)[/green]\n")
        
        return {"bytes": len(self.document.raw_html)}

    def _run_stage_parse(self) -> Dict[str, Any]:
        """Stage 2: Parse HTML into sections."""
        console.print("[bold yellow]▶ STAGE 2: Parsing document...[/bold yellow]")
        start = time.time()
        
        parser = SECParser(self.config)
        self.sections = parser.parse(self.document.raw_html)
        
        self.stage_times["parse"] = time.time() - start
        console.print(f"  [green]✓ Parsed {len(self.sections)} sections "
                     f"({self.stage_times['parse']:.1f}s)[/green]\n")
        
        return {"sections": len(self.sections)}

    def _run_stage_chunk(self) -> Dict[str, Any]:
        """Stage 3: Chunk sections into LLM-friendly pieces."""
        console.print("[bold yellow]▶ STAGE 3: Chunking document...[/bold yellow]")
        start = time.time()
        
        chunker = DocumentChunker(self.config)
        self.chunks = chunker.chunk_sections(self.sections)
        
        # Update stats
        self.stats.total_chunks = len(self.chunks)
        for chunk in self.chunks:
            section = chunk.item
            self.stats.by_section[section] = self.stats.by_section.get(section, 0) + 1
        
        self.stage_times["chunk"] = time.time() - start
        console.print(f"  [green]✓ Created {len(self.chunks)} chunks "
                     f"({self.stage_times['chunk']:.1f}s)[/green]\n")
        
        return {"chunks": len(self.chunks)}

    def _run_stage_generate(self, target: int = 120) -> Dict[str, Any]:
        """Stage 4: Generate Q&A pairs."""
        console.print(f"[bold yellow]▶ STAGE 4: Generating Q&A pairs (target={target})...[/bold yellow]")
        start = time.time()
        
        generator = QAGenerator(self.config)
        self.raw_pairs = generator.generate_from_chunks(self.chunks, target=target)
        
        # Update stats
        self.stats.total_generated = len(self.raw_pairs)
        for pair in self.raw_pairs:
            qtype = pair.question_type
            diff = pair.difficulty
            self.stats.by_type[qtype] = self.stats.by_type.get(qtype, 0) + 1
            self.stats.by_difficulty[diff] = self.stats.by_difficulty.get(diff, 0) + 1
        
        self.stage_times["generate"] = time.time() - start
        console.print(f"  [green]✓ Generated {len(self.raw_pairs)} pairs "
                     f"({self.stage_times['generate']:.1f}s)[/green]\n")
        
        return {"generated": len(self.raw_pairs)}

    def _run_stage_verify(self) -> Dict[str, Any]:
        """Stage 5: Verify Q&A pairs."""
        console.print("[bold yellow]▶ STAGE 5: Verifying Q&A pairs...[/bold yellow]")
        start = time.time()
        
        verifier = AnswerVerifier(self.config, self.chunks)
        self.verified_pairs = verifier.verify_all(self.raw_pairs)
        
        # Update stats
        self.stats.total_verified = len(self.verified_pairs)
        self.stats.total_rejected = len(self.raw_pairs) - len(self.verified_pairs)
        self.stats.verification_pass_rate = (
            len(self.verified_pairs) / max(len(self.raw_pairs), 1)
        )
        
        self.stage_times["verify"] = time.time() - start
        console.print(f"  [green]✓ Verified {len(self.verified_pairs)}/{len(self.raw_pairs)} pairs "
                     f"({self.stage_times['verify']:.1f}s)[/green]\n")
        
        return {
            "verified": len(self.verified_pairs),
            "rejected": self.stats.total_rejected,
            "pass_rate": self.stats.verification_pass_rate,
        }

    def _run_stage_output(self) -> Dict[str, str]:
        """Stage 6: Write output files."""
        console.print("[bold yellow]▶ STAGE 6: Writing output files...[/bold yellow]")
        start = time.time()
        
        # Ensure output directory exists
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Build output object
        filing_config = self.config.get("filing", {})
        output = PipelineOutput(
            metadata={
                "pipeline_version": "1.0.0",
                "config_file": self.config_path,
                "run_duration_seconds": time.time() - self.start_time,
                "stage_times": self.stage_times,
            },
            filing=FilingMetadata(
                company=filing_config.get("company", "Microsoft Corporation"),
                cik=filing_config.get("cik", "0000789019"),
                filing_type="10-K",
                fiscal_year_end=str(filing_config.get("fiscal_year", 2025)),
                accession_number=filing_config.get("accession_number", ""),
                sec_url=filing_config.get("url", ""),
            ),
            statistics=self.stats,
            qa_pairs=self.verified_pairs,
            generator_model=self.config.get("generator", {}).get("model", ""),
            verifier_model=self.config.get("verifier", {}).get("model", ""),
        )
        
        # Write JSON
        json_path = output_dir / "qa_pairs.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output.model_dump(mode="json"), f, indent=2, default=str)
        console.print(f"  [green]✓ JSON: {json_path}[/green]")
        
        # Write CSV
        csv_path = output_dir / "qa_pairs.csv"
        rows = output.to_csv_rows()
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False, encoding="utf-8")
            console.print(f"  [green]✓ CSV:  {csv_path}[/green]")
        
        # Write statistics log
        log_path = output_dir / "pipeline_log.json"
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": self.config_path,
            "statistics": self.stats.model_dump(),
            "stage_times": self.stage_times,
            "total_duration": time.time() - self.start_time,
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        console.print(f"  [green]✓ Log:  {log_path}[/green]")
        
        self.stage_times["output"] = time.time() - start
        console.print()
        
        return {
            "json": str(json_path),
            "csv": str(csv_path),
            "log": str(log_path),
        }

    # ================================================================ #
    # SUMMARY & REPORTING
    # ================================================================ #
    def _print_final_summary(self):
        """Print final pipeline summary."""
        total_time = time.time() - self.start_time
        
        console.print("[bold cyan]╔══════════════════════════════════════════════════════╗[/bold cyan]")
        console.print("[bold cyan]║              PIPELINE COMPLETE                       ║[/bold cyan]")
        console.print("[bold cyan]╚══════════════════════════════════════════════════════╝[/bold cyan]\n")
        
        # Results table
        table = Table(title="Pipeline Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        table.add_row("Total Chunks", str(self.stats.total_chunks))
        table.add_row("Q&A Generated", str(self.stats.total_generated))
        table.add_row("Q&A Verified", str(self.stats.total_verified))
        table.add_row("Q&A Rejected", str(self.stats.total_rejected))
        table.add_row("Pass Rate", f"{self.stats.verification_pass_rate:.1%}")
        table.add_row("Total Time", f"{total_time:.1f}s")
        
        console.print(table)
        
        # Distribution tables
        if self.stats.by_type:
            type_table = Table(title="\nBy Question Type", show_header=True)
            type_table.add_column("Type", style="cyan")
            type_table.add_column("Count", justify="right")
            for qtype, count in sorted(self.stats.by_type.items()):
                type_table.add_row(qtype, str(count))
            console.print(type_table)
        
        if self.stats.by_difficulty:
            diff_table = Table(title="\nBy Difficulty", show_header=True)
            diff_table.add_column("Difficulty", style="cyan")
            diff_table.add_column("Count", justify="right")
            for diff, count in sorted(self.stats.by_difficulty.items()):
                diff_table.add_row(diff, str(count))
            console.print(diff_table)
        
        # Timing breakdown
        console.print("\n[bold]Stage Timing:[/bold]")
        for stage, duration in self.stage_times.items():
            console.print(f"  {stage:12s}: {duration:6.1f}s")
        
        console.print(f"\n[bold green]✅ Output saved to output/ directory[/bold green]\n")

    def get_statistics(self) -> PipelineStats:
        """Get current pipeline statistics."""
        return self.stats

    def get_verified_pairs(self) -> List[VerifiedQAPair]:
        """Get list of verified Q&A pairs."""
        return self.verified_pairs