"""Evaluate Java token removal on paired docstring-to-code retrieval."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

from ast_simplifier import JavaCodeSimplifier
from model_search import CodeSearchModel


MODELS = {"codebert": "microsoft/codebert-base", "codet5": "Salesforce/codet5-base"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="codet5")
    parser.add_argument("--model-path", help="Local checkpoint directory or Hugging Face model ID")
    parser.add_argument("--train-file", type=Path, default=Path("java_research/train.txt"))
    parser.add_argument("--test-file", type=Path, default=Path("java_research/java_test_0.jsonl"))
    parser.add_argument("--ratios", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--sample-fraction", type=float, default=0.1,
                        help="Fraction of usable pairs sampled independently from each split")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/search"))
    args = parser.parse_args()
    if any(not 0 <= r <= 1 for r in args.ratios) or not 0 < args.sample_fraction <= 1:
        parser.error("ratios must be in [0, 1] and sample-fraction in (0, 1]")
    if len(set(args.ratios)) != len(args.ratios):
        parser.error("each ratio must be unique")
    if args.epochs < 1 or args.batch_size < 2 or args.seed < 0:
        parser.error("epochs must be positive, batch-size at least 2, and seed nonnegative")
    return args


def training_pairs(path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("<CODESPLIT>", 4)
            if len(fields) != 5 or fields[0].strip() not in {"0", "1"}:
                raise ValueError(f"{path}:{line_number}: expected five <CODESPLIT> fields")
            label, _, _, docstring, code = fields
            if label.strip() == "1" and code.strip() and docstring.strip():
                yield {"code": code, "docstring": docstring.strip()}


def test_pairs(path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                code = item.get("code") or item.get("code_tokens") or item.get("function_tokens")
                doc = item.get("docstring") or item.get("docstring_tokens")
                if isinstance(code, list):
                    code = " ".join(code)
                if isinstance(doc, list):
                    doc = " ".join(doc)
                if not isinstance(code, str) or not isinstance(doc, str):
                    raise ValueError("missing code or docstring")
            except (ValueError, AttributeError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid retrieval record: {error}") from error
            if code.strip() and doc.strip():
                yield {"code": code, "docstring": doc.strip().splitlines()[0]}


def sample_pairs(reader, path, fraction, seed):
    # Choose the subset before materializing code strings from the full corpus.
    total = sum(1 for _ in reader(path))
    size = max(1, int(total * fraction))
    if size < 2 or total < 2:
        raise ValueError(f"{path}: need at least two sampled pairs (usable rows: {total})")
    rng = np.random.default_rng(seed)
    indices = set(rng.choice(total, size=size, replace=False))
    return Dataset.from_list([row for index, row in enumerate(reader(path)) if index in indices])


def evaluate_mrr(model, loader, device):
    """Rank the single aligned code row for each query against the entire sampled pool."""
    model.eval()
    codes, queries = [], []
    with torch.inference_mode():
        for code, query in loader:
            codes.append(model.get_embeddings(**code.to(device)).cpu().numpy())
            queries.append(model.get_embeddings(**query.to(device)).cpu().numpy())
    codes = np.concatenate(codes)
    queries = np.concatenate(queries)
    # A stable sort resolves exact score ties in candidate-file order.
    reciprocal_ranks = []
    for start in range(0, len(queries), 128):
        scores = queries[start:start + 128] @ codes.T
        for index, row in enumerate(scores, start=start):
            order = np.argsort(-row, kind="stable")
            rank = np.flatnonzero(order == index)[0] + 1
            reciprocal_ranks.append(1.0 / rank)
    return float(np.mean(reciprocal_ranks))


def run_single_experiment(args, ratio, train, test, model_path, tokenizer, device):
    # Reset before each ratio so initialization, dropout and shuffling use the same seed.
    set_seed(args.seed)
    simplifier = JavaCodeSimplifier()

    def simplify(row):
        return {"code": simplifier.simplify(row["code"], remove_ratio=ratio)}

    def collate(rows):
        options = dict(padding=True, truncation=True, max_length=256, return_tensors="pt",
                       return_token_type_ids=False)
        code = tokenizer([row["code"] for row in rows], **options)
        query = tokenizer([row["docstring"] for row in rows], **options)
        return code, query

    train_loader = DataLoader(train.map(simplify, load_from_cache_file=False),
                              batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test.map(simplify, load_from_cache_file=False),
                             batch_size=args.batch_size, collate_fn=collate)
    model = CodeSearchModel(model_path, model_type=args.model).to(device)
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(args.epochs):
        for code, query in tqdm(train_loader, desc=f"Ratio {ratio:g} | epoch {epoch + 1}"):
            optimizer.zero_grad()
            loss = model(code.to(device), query.to(device))
            loss.backward()
            optimizer.step()
    return evaluate_mrr(model, test_loader, device)


def main():
    args = parse_args()
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise SystemExit(f"Output is not an empty directory: {args.output_dir}. Choose a new --output-dir.")
    train = sample_pairs(training_pairs, args.train_file, args.sample_fraction, args.seed)
    test = sample_pairs(test_pairs, args.test_file, args.sample_fraction, args.seed)
    local = Path(f"{args.model}-base-local")
    model_path = args.model_path or (str(local) if local.is_dir() else MODELS[args.model])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(train)} training pairs, {len(test)} test candidates, device={device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with (args.output_dir / "results.csv").open("x", newline="", encoding="utf-8") as dest:
        fields = ["model", "model_path", "ratio", "test_mrr", "train_pairs", "test_pairs", "seed",
                  "sample_fraction", "epochs", "batch_size", "train_file", "test_file"]
        writer = csv.DictWriter(dest, fieldnames=fields)
        writer.writeheader()
        for ratio in args.ratios:
            mrr = run_single_experiment(args, ratio, train, test, model_path, tokenizer, device)
            print(f"ratio={ratio:g} | test MRR={mrr:.6f}")
            row = dict(model=args.model, model_path=model_path, ratio=ratio, test_mrr=mrr,
                       train_pairs=len(train), test_pairs=len(test), seed=args.seed,
                       sample_fraction=args.sample_fraction, epochs=args.epochs,
                       batch_size=args.batch_size, train_file=args.train_file, test_file=args.test_file)
            results.append(row)
            writer.writerow(row)
            dest.flush()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(args.ratios, [row["test_mrr"] for row in results], marker="o", label=args.model)
    ax.set(xlabel="Token removal ratio", ylabel="Test MRR", title="Java code search")
    ax.legend()
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(args.output_dir / "search_benchmark_result.png")
    plt.close(fig)
    print(f"Saved results.csv and search_benchmark_result.png to {args.output_dir}")


if __name__ == "__main__":
    main()
