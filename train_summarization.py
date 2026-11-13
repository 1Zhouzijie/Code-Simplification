"""Fine-tune CodeT5 to generate Java method summaries from simplified tokens."""

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sacrebleu import corpus_bleu
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from ast_simplifier import JavaCodeSimplifier


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Local checkpoint directory or Hugging Face model ID")
    parser.add_argument("--data-dir", type=Path, default=Path("java"),
                        help="Directory containing train.jsonl, valid.jsonl and test.jsonl")
    parser.add_argument("--ratio", type=float, default=0, help="Fraction of Java tokens to remove")
    parser.add_argument("--train-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, help="New run directory for checkpoints, metrics and model")
    args = parser.parse_args()
    if not 0 <= args.ratio <= 1 or not 0 < args.train_fraction <= 1:
        parser.error("ratio must be in [0, 1] and train-fraction in (0, 1]")
    if args.epochs < 1 or args.batch_size < 1 or args.seed < 0:
        parser.error("epochs and batch-size must be positive, and seed nonnegative")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    model_path = args.model_path or (
        "codet5-base-local" if Path("codet5-base-local").is_dir() else "Salesforce/codet5-base"
    )
    run_dir = args.output_dir or Path("outputs/summarization") / f"ratio-{args.ratio:g}"
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise SystemExit(f"Output is not an empty directory: {run_dir}. Choose a new --output-dir.")
    output_dir = run_dir / "model"
    checkpoint_dir = run_dir / "checkpoints"
    files = {"train": "train.jsonl", "validation": "valid.jsonl", "test": "test.jsonl"}
    dataset = load_dataset("json", data_files={key: str(args.data_dir / name) for key, name in files.items()})
    for split, data in dataset.items():
        if not len(data) or not {"code", "docstring"}.issubset(data.column_names):
            raise ValueError(f"{files[split]} must contain nonempty records with code and docstring fields")
    train = dataset["train"]
    if args.train_fraction < 1:
        count = max(1, int(len(train) * args.train_fraction))
        dataset["train"] = train.shuffle(seed=args.seed).select(range(count))
    print(" | ".join(f"{split}={len(data)}" for split, data in dataset.items()))

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    simplifier = JavaCodeSimplifier()

    def text_or_default(value, default):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return str(value).strip() or default

    def preprocess(examples):
        codes = [simplifier.simplify(text_or_default(code, ""), args.ratio) or "empty code"
                 for code in examples["code"]]
        summaries = [text_or_default(doc, "No documentation") for doc in examples["docstring"]]
        inputs = tokenizer(codes, max_length=512, truncation=True)
        labels = tokenizer(text_target=summaries, max_length=128, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        # Added tokens are valid too. Batch-gather padding and out-of-range IDs cannot be decoded.
        predictions = np.where((predictions >= 0) & (predictions < len(tokenizer)),
                               predictions, tokenizer.pad_token_id)
        labels = np.where(labels >= 0, labels, tokenizer.pad_token_id)
        hypotheses = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {"bleu": corpus_bleu([s.strip() for s in hypotheses],
                                    [[s.strip() for s in references]]).score}

    encoded = {
        split: data.map(preprocess, batched=True, batch_size=100, remove_columns=data.column_names,
                        desc=f"Tokenizing {split}")
        for split, data in dataset.items()
    }
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        generation_max_length=128,
        fp16=cuda and not bf16,
        bf16=bf16,
        logging_steps=50,
        gradient_accumulation_steps=4,
        dataloader_num_workers=0,
        dataloader_pin_memory=cuda,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        gradient_checkpointing=True,
        seed=args.seed,
        report_to="none",
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    trainer.remove_callback(EarlyStoppingCallback)
    metrics = trainer.evaluate(eval_dataset=encoded["test"], metric_key_prefix="test")
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
