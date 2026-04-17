"""
Unified LoRA/QLoRA training script using Unsloth.
Config-driven to train any of the specialist SLMs.
"""
import argparse
from pathlib import Path
import structlog
import yaml

logger = structlog.get_logger(__name__)

def main(config_path: str) -> None:
    logger.info("train_lora.start", config=config_path)
    
    # Check if unsloth is available (optional dependency)
    try:
        from unsloth import FastLanguageModel
        import torch
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
    except ImportError:
         logger.error("train_lora.missing_deps", msg="Please install the optional [fine-tune] dependencies.")
         return

    with open(config_path, "r") as f:
         cfg = yaml.safe_load(f)

    max_seq_length = cfg.get("max_seq_length", 2048)
    dtype = None # Auto
    load_in_4bit = cfg.get("load_in_4bit", True)
    
    # 1. Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = cfg["model_name"],
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    # 2. Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = cfg.get("lora_r", 16),
        target_modules = cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj",
                                                     "gate_proj", "up_proj", "down_proj"]),
        lora_alpha = cfg.get("lora_alpha", 16),
        lora_dropout = 0, # Optimization
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )
    
    # 3. Load dataset
    dataset = load_dataset("json", data_files=cfg["dataset_path"], split="train")

    def format_chatml(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    
    dataset = dataset.map(format_chatml)

    # 4. Train
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make this True for short context
        args = TrainingArguments(
            per_device_train_batch_size = cfg.get("batch_size", 2),
            gradient_accumulation_steps = cfg.get("grad_accum", 4),
            warmup_steps = 5,
            num_train_epochs = cfg.get("epochs", 1),
            learning_rate = float(cfg.get("learning_rate", 2e-4)),
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = cfg["output_dir"],
        ),
    )
    
    logger.info("train_lora.training")
    trainer_stats = trainer.train()
    
    # 5. Save LoRA weights
    model.save_pretrained(cfg["output_dir"]) # Local saving
    logger.info("train_lora.done", stats=str(trainer_stats))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()
    main(args.config)
