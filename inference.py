import argparse
from functools import lru_cache
from pathlib import Path

from ast_simplifier import JavaCodeSimplifier


@lru_cache(maxsize=1)
def _load_checkpoint(model_path, tokenizer_path):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def default_checkpoint(remove_ratio):
    run_model = Path("outputs/summarization") / f"ratio-{remove_ratio:g}" / "model"
    legacy_model = Path(f"final_model_{remove_ratio:g}")
    if not run_model.is_dir() and legacy_model.is_dir():
        return legacy_model
    return run_model


def generate_summary(original_code, remove_ratio=0.3, model_path=None, tokenizer_path=None):
    """Generate a summary using the checkpoint trained at the requested removal ratio."""
    if not 0 <= remove_ratio <= 1:
        raise ValueError("remove_ratio must be between 0 and 1")
    if model_path is None:
        model_path = str(default_checkpoint(remove_ratio))
    tokenizer, model = _load_checkpoint(model_path, tokenizer_path)
    simplified = JavaCodeSimplifier().simplify(original_code, remove_ratio)
    inputs = tokenizer(simplified, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate a summary for a Java method.")
    parser.add_argument("file", nargs="?", type=Path, help="UTF-8 Java file (default: built-in example)")
    parser.add_argument("--ratio", type=float, default=0.3, help="Use the same removal ratio as training")
    parser.add_argument("--model-path", help="Override the fine-tuned checkpoint for this ratio")
    parser.add_argument("--tokenizer-path", help="Override the tokenizer saved with the checkpoint")
    args = parser.parse_args()
    if not 0 <= args.ratio <= 1:
        parser.error("--ratio must be between 0 and 1")
    if args.model_path is None and not default_checkpoint(args.ratio).is_dir():
        parser.error("train a checkpoint at this ratio first, or provide --model-path")
    source = "public int calculateSum(int a, int b) { return a + b; }"
    if args.file:
        try:
            source = args.file.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            parser.error(str(error))
    print(generate_summary(source, args.ratio, args.model_path, args.tokenizer_path))


if __name__ == "__main__":
    main()
