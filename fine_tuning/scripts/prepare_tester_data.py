"""
Prepare Tester dataset.
Maps function signatures + docstrings to Pytest suite generation.
"""
import argparse
import json
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

def main(input_dir: str, output_jsonl: str) -> None:
    logger.info("prepare_tester.start", input=input_dir)
    in_path = Path(input_dir)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # MVP Skeleton
    logger.info("prepare_tester.done", count=0, output=str(out_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()
    main(args.input, args.output)
