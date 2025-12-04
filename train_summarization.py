import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from datasets import load_dataset
import evaluate
import numpy as np
from ast_simplifier import JavaCodeSimplifier
import math
import os

# ================= 性能优化配置 =================
# Windows 上最稳妥的方式是设为 0
NUM_WORKERS = 0

# ================= 全局配置 =================
MODEL_CHECKPOINT = "./codet5-base-local"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
SIMPLIFY_RATIO = 0.3


TRAIN_FRACTION = 1/3

# ================= 全局对象初始化 =================
# 注意：在 Windows 多进程中，这些全局对象可能会被重复初始化，
# 但由于 NUM_WORKERS=0，这里是安全的。
simplifier = JavaCodeSimplifier()
# 预先加载 tokenizer 以供全局函数使用
try:
    tokenizer_global = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False)
except:
    tokenizer_global = None  # 还没下载好时可能报错，先忽略


# ================= 数据预处理函数 (全局) =================
def preprocess_data(examples):
    # 如果 tokenizer 还没加载（比如第一次运行），尝试加载
    global tokenizer_global
    if tokenizer_global is None:
        tokenizer_global = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False)

    tokenizer = tokenizer_global

    # 1. 统一转成列表
    codes = examples['code']
    docstrings = examples['docstring']

    # 2. 安全字符串转换函数
    def safe_str(x):
        if x is None:
            return ""
        if isinstance(x, (float, int)) and math.isnan(float(x)):
            return ""
        s = str(x)
        try:
            # 忽略非法字符
            s = s.encode('utf-8', 'ignore').decode('utf-8')
        except Exception:
            return ""
        return s.strip()

    # 3. 预处理列表
    if not isinstance(codes, list):
        codes = [codes]
    if not isinstance(docstrings, list):
        docstrings = [docstrings]

    clean_codes = [safe_str(c) for c in codes]
    clean_docs = [safe_str(d) for d in docstrings]

    # 4. 简化代码
    final_codes = []
    for c in clean_codes:
        if not c:
            final_codes.append("empty code")
            continue
        try:
            # 调用简化器
            result = simplifier.simplify(c, SIMPLIFY_RATIO)
            safe_result = safe_str(result)
            final_codes.append(safe_result if safe_result else "empty code")
        except Exception:
            final_codes.append("empty code")

    # 5. 处理摘要
    final_docs = [d if d else "No documentation" for d in clean_docs]

    # 6. Tokenize
    # 处理空数据情况
    if len(final_codes) == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    model_inputs = tokenizer(
        text=final_codes,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text=final_docs,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ================= 主执行块 =================
if __name__ == "__main__":
    # 设置 Hugging Face 镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    print(f"Is CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        BF16_SUPPORTED = torch.cuda.get_device_capability()[0] >= 8
    else:
        BF16_SUPPORTED = False

    # 1. 重新确保 tokenizer 可用
    if tokenizer_global is None:
        tokenizer_global = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False)
    tokenizer = tokenizer_global

    # 定义 compute_metrics (闭包引用 tokenizer)
    bleu = evaluate.load("sacrebleu")


    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # 处理 -100
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}


    # 2. 加载数据
    data_files = {"train": "java/train.jsonl", "test": "java/test.jsonl"}
    dataset = load_dataset("json", data_files=data_files)

    dataset['train'] = dataset['train'].train_test_split(train_size=TRAIN_FRACTION, shuffle=True, seed=42)['train']
    print(f"Train size after sampling: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    # 调试：只取少量数据测试流程！(跑通后再注释掉)
    # print("Debug: Taking subset of data...")
    # dataset['train'] = dataset['train'].select(range(50))
    # dataset['test'] = dataset['test'].select(range(10))

    # 3. 处理数据
    print("Processing dataset...")
    column_names = dataset['train'].column_names

    tokenized_datasets = dataset.map(
        preprocess_data,
        batched=True,
        batch_size=100,
        remove_columns=column_names,  # 强制移除旧列
        desc="Processing dataset",
        num_proc=NUM_WORKERS if NUM_WORKERS > 0 else None,
        load_from_cache_file=False  # 禁用缓存，防止读取到以前错误的空结果
    )

    print(f"Processed columns: {tokenized_datasets['train'].column_names}")

    # 强制格式检查
    model_columns = ["input_ids", "attention_mask", "labels"]
    tokenized_datasets.set_format(
        type="torch",
        columns=model_columns,
        output_all_columns=False
    )

    # 4. 模型初始化
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # 5. 参数设置
    args = Seq2SeqTrainingArguments(
        output_dir=f"./codet5-simplified-{SIMPLIFY_RATIO}",
        eval_strategy="epoch",
        save_strategy="epoch",  # 必须匹配
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        fp16=not BF16_SUPPORTED,
        bf16=BF16_SUPPORTED,
        push_to_hub=False,
        logging_steps=50,
        gradient_accumulation_steps=4,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        gradient_checkpointing=True,
        include_inputs_for_metrics=False,
        remove_unused_columns=False,  # 防止误删列
    )

    # 6. 开始训练
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(f"./final_model_{SIMPLIFY_RATIO}")