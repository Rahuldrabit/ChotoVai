"""
IMAGINE Trainer — fine-tune an SLM on adversarial debate traces.

Implements the 'Agentic Training' step of the IMAGINE framework:
  1. Load ShareGPT-format debate traces from DebateTraceCollector
  2. Convert to instruction-following format
  3. Fine-tune the target SLM using LoRA (via Unsloth + TRL)

The goal: one small model internalises the structured debate logic of the
entire Coder+Critic team, removing the need for two model calls at inference.

Dependencies: unsloth, trl, datasets (optional group — install with [fine-tune])
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ImagineTrainer:
    """
    Fine-tunes an SLM on debate trace data using LoRA via Unsloth.

    Usage:
        trainer = ImagineTrainer(base_model="Qwen/Qwen2.5-Coder-7B-Instruct")
        dataset = trainer.prepare_dataset(Path("data/debate_traces/sft_export.jsonl"))
        trainer.train(dataset, output_dir=Path("models/debate-distilled-v1"))
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        max_seq_length: int = 8192,
        load_in_4bit: bool = True,
    ) -> None:
        self.base_model = base_model
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit

    # ──────────────────────────────────────────────────────────────────────
    # Dataset preparation
    # ──────────────────────────────────────────────────────────────────────

    def prepare_dataset(self, traces_path: Path | str) -> Any:
        """
        Load ShareGPT-format JSONL debate traces and convert them to a
        HuggingFace Dataset with the chat-template instruction format.

        Args:
            traces_path: Path to JSONL file exported by DebateTraceCollector.

        Returns:
            HuggingFace Dataset ready for SFT training.
        """
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError(
                "datasets not installed. Run: pip install 'slm-agent[fine-tune]'"
            ) from exc

        traces_path = Path(traces_path)
        if not traces_path.exists():
            raise FileNotFoundError(f"Traces file not found: {traces_path}")

        rows: list[dict[str, Any]] = []
        with traces_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        logger.info("imagine_trainer.loading_traces", count=len(rows), path=str(traces_path))

        # Filter: only use winning debates (CODER_WINS) and escalated (frontier
        # resolution) as high-quality positive examples. Include DEADLOCK as
        # negative examples with lower weight.
        winning = [r for r in rows if r.get("outcome") in ("coder_wins", "escalated")]
        deadlocked = [r for r in rows if r.get("outcome") == "deadlock"]

        logger.info(
            "imagine_trainer.dataset_composition",
            winning=len(winning),
            deadlocked=len(deadlocked),
        )

        # Convert to instruction format
        examples = [self._to_instruction(r, is_positive=True) for r in winning]
        examples += [self._to_instruction(r, is_positive=False) for r in deadlocked]

        # Filter None (malformed traces)
        examples = [e for e in examples if e is not None]

        dataset = Dataset.from_list(examples)
        logger.info("imagine_trainer.dataset_ready", size=len(dataset))
        return dataset

    def _to_instruction(
        self, trace: dict[str, Any], is_positive: bool
    ) -> dict[str, Any] | None:
        """Convert one trace to the chat-template format expected by TRL SFTTrainer."""
        conversations = trace.get("conversations", [])
        if not conversations:
            return None

        system = trace.get("system", "")
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]

        for turn in conversations:
            role = "assistant" if turn["from"] == "gpt" else "user"
            messages.append({"role": role, "content": turn["value"]})

        # For negative examples, append a correction note as the last assistant turn
        if not is_positive and trace.get("final_code"):
            correction = (
                "[IMPROVEMENT NEEDED]\n"
                "The code above did not pass adversarial review. "
                f"Final critic score: {trace.get('final_score', 0)}/10. "
                "Key issues must be addressed before merging."
            )
            messages.append({"role": "assistant", "content": correction})

        return {
            "messages": messages,
            "outcome": trace.get("outcome", "unknown"),
            "final_score": trace.get("final_score", 0),
            "is_positive": is_positive,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def train(
        self,
        dataset: Any,
        output_dir: Path | str,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        per_device_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
    ) -> None:
        """
        Fine-tune the base model on debate traces using LoRA via Unsloth + TRL.

        Args:
            dataset:             HuggingFace Dataset from prepare_dataset().
            output_dir:          Where to save the LoRA adapter weights.
            lora_rank:           LoRA rank (r). Higher = more capacity, more VRAM.
            lora_alpha:          LoRA scaling factor (usually 2*rank).
            learning_rate:       AdamW learning rate.
            num_epochs:          Number of full dataset passes.
            per_device_batch_size: Micro-batch size.
            gradient_accumulation_steps: Effective batch = micro * accum.
        """
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer, SFTConfig
        except ImportError as exc:
            raise ImportError(
                "unsloth/trl not installed. Run: pip install 'slm-agent[fine-tune]'"
            ) from exc

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "imagine_trainer.loading_base_model",
            model=self.base_model,
            load_in_4bit=self.load_in_4bit,
        )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            dtype=None,  # Auto-detect
        )

        # Apply LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        training_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_steps=10,
            bf16=True,
            max_seq_length=self.max_seq_length,
            dataset_text_field=None,  # We use messages
            packing=False,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )

        logger.info("imagine_trainer.training_start", epochs=num_epochs)
        trainer.train()

        # Save LoRA adapter only (not full weights — much smaller)
        model.save_pretrained(str(output_dir / "lora_adapter"))
        tokenizer.save_pretrained(str(output_dir / "lora_adapter"))

        logger.info("imagine_trainer.done", output_dir=str(output_dir))
