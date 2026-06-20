#!/usr/bin/env python3
"""
10-K Q&A Pipeline - Main Entry Point

Usage:
    python main.py                      # Run full pipeline (120 pairs)
    python main.py --target 50          # Generate 50 pairs
    python main.py --stage fetch        # Run specific stage
    python main.py --debug              # Enable debug logging
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import QAPipeline
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="10-K Q&A Pipeline - Generate verified Q&A pairs from SEC filings"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=120,
        help="Target number of Q&A pairs to generate (default: 120)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["fetch", "parse", "chunk", "generate", "verify", "output", "all"],
        default="all",
        help="Pipeline stage to run (default: all)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    try:
        pipeline = QAPipeline(config_path=args.config, debug=args.debug)
        
        if args.stage == "all":
            results = pipeline.run(target_pairs=args.target)
        else:
            results = pipeline.run_stage(args.stage)
        
        if results.get("success", True):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()