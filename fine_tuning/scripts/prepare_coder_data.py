"""
Prepare Coder dataset.
Converts synthetic agent trajectories or git commits into (ContextPack, Diff) training pairs.
"""
import argparse
import json
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

def main(input_dir: str, output_jsonl: str) -> None:
    logger.info("prepare_coder.start", input=input_dir)
    in_path = Path(input_dir)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        # In MVP, this is a skeleton that would parse episodic JSON logs
        # and format them into ChatML blocks for Qwen/Llama fine-tuning.
        for log in in_path.glob("*.json"):
            try:
                data = json.loads(log.read_text())
                prompt = data.get("context_pack_text", "") + "\nTask: " + data.get("task", "")
                response = data.get("unified_diff", "")
                
                # ChatML format
                record = {
                    "messages": [
                        {"role": "system", "content": "You are a precise coder. Emit a unified diff."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                }
                f.write(json.dumps(record) + "\n")
                count += 1
            except Exception as e:
                logger.warning("prepare_coder.skip", file=log.name, error=str(e))
                
    logger.info("prepare_coder.done", count=count, output=str(out_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()
    main(args.input, args.output)
