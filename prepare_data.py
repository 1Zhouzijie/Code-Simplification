import argparse

from datasets import load_dataset

from ast_simplifier import JavaCodeSimplifier


simplifier = JavaCodeSimplifier()
REMOVE_RATIO = 0.3


def preprocess_function(examples, remove_ratio=REMOVE_RATIO):
    return {
        "simplified_code": [simplifier.simplify(code, remove_ratio) for code in examples["code"]],
        "summary": examples["docstring"],
    }


def load_and_process_data(file_path, limit=1000, remove_ratio=REMOVE_RATIO):
    """Preview up to limit JSONL records with code and docstring fields."""
    if limit < 1 or not 0 <= remove_ratio <= 1:
        raise ValueError("limit must be positive and remove_ratio must be between 0 and 1")
    dataset = load_dataset("json", data_files=str(file_path), split="train")
    dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset.map(preprocess_function, batched=True, fn_kwargs={"remove_ratio": remove_ratio})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview Java summary dataset preprocessing.")
    parser.add_argument("file", nargs="?", default="java/train.jsonl")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--ratio", type=float, default=REMOVE_RATIO)
    args = parser.parse_args()
    if args.limit < 1 or not 0 <= args.ratio <= 1:
        parser.error("--limit must be positive and --ratio must be between 0 and 1")
    processed = load_and_process_data(args.file, args.limit, args.ratio)
    if len(processed):
        print(processed[0])
