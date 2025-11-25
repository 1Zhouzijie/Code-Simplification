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
# 设置多进程数据加载的工作进程数
NUM_WORKERS = min(4, os.cpu_count() or 1)

print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # 检测是否支持 bf16 (Ampere 架构及以上)
    BF16_SUPPORTED = torch.cuda.get_device_capability()[0] >= 8
    print(f"BF16 supported: {BF16_SUPPORTED}")
else:
    BF16_SUPPORTED = False
    print("CUDA not available, running on CPU")


# ================= 配置 =================
MODEL_CHECKPOINT = "./codet5-base-local"
MAX_INPUT_LENGTH = 512  # CodeT5 输入限制
MAX_TARGET_LENGTH = 128  # 摘要输出限制
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
SIMPLIFY_RATIO = 0.3  # 30% 简化率

# ================= 数据准备 =================
# 加载你的数据集
data_files = {"train": "java/train.jsonl", "test": "java/test.jsonl"}
dataset = load_dataset("json", data_files=data_files)

# 修复点 1: 强制使用 use_fast=False，避免 Rust Tokenizer 的类型报错
# CodeT5 基于 Roberta，Python 版 tokenizer 更加稳定
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False)
simplifier = JavaCodeSimplifier()
bleu = evaluate.load("sacrebleu")


def preprocess_data(examples):
    # 1. 统一转成列表
    codes = examples['code']
    docstrings = examples['docstring']

    # 2. 加强版的安全字符串转换函数
    def safe_str(x):
        if x is None:
            return ""
        if isinstance(x, (float, int)) and math.isnan(float(x)):
            return ""

        s = str(x)

        # 核心修复: 尝试编码为 utf-8 (忽略错误) 再解码，彻底去除非法代理字符
        try:
            # 'surrogatepass' 允许代理字符存在，但我们这里想去掉它们，或者转成 replacement char
            # 方法 A: 忽略错误 (直接丢弃非法字符)
            s = s.encode('utf-8', 'ignore').decode('utf-8')

            # 或者 方法 B: 替换错误 (变成 ) - 推荐忽略，以免影响代码语法
            # s = s.encode('utf-8', 'replace').decode('utf-8')
        except Exception:
            # 兜底: 如果连 encode 都报错，就强制过滤
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
            result = simplifier.simplify(c, SIMPLIFY_RATIO)
            # 对简化后的结果再次清洗，以防万一
            safe_result = safe_str(result)
            final_codes.append(safe_result if safe_result else "empty code")
        except Exception as e:
            # 打印出错的代码片段以便调试 (可选)
            # print(f"Simplification error on: {c[:20]}... : {e}")
            final_codes.append("empty code")

    # 5. 处理摘要
    final_docs = [d if d else "No documentation" for d in clean_docs]

    # 6. Tokenize
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


# 处理数据集
# 使用多进程加速数据预处理
tokenized_datasets = dataset.map(
    preprocess_data,
    batched=True,
    batch_size=100,
    remove_columns=dataset['train'].column_names,
    desc="Processing dataset",
    num_proc=NUM_WORKERS,  # 多进程加速数据预处理
)

# ================= 模型设置 =================
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

# 使用 torch.compile 加速模型 (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    print("Enabling torch.compile for faster training...")
    model = torch.compile(model)


# 定义评估指标计算函数
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


# ================= 训练参数 =================
args = Seq2SeqTrainingArguments(
    output_dir=f"./codet5-simplified-{SIMPLIFY_RATIO}",
    eval_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=EPOCHS,
    predict_with_generate=True,
    # 混合精度训练: 优先使用 bf16 (更稳定), 否则使用 fp16
    fp16=not BF16_SUPPORTED,
    bf16=BF16_SUPPORTED,
    push_to_hub=False,
    logging_steps=50,
    gradient_accumulation_steps=4,
    # 数据加载优化
    dataloader_num_workers=NUM_WORKERS,
    dataloader_pin_memory=True,
    # 早停和最佳模型保存
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    # 梯度检查点已在模型上启用
    gradient_checkpointing=True,
    # 禁用完整的 eval 预测以加速
    include_inputs_for_metrics=False,
)

# 数据整理器
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 早停回调: 如果验证集指标连续 2 个 epoch 没有改善，则提前停止训练
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

# ================= 开始训练 =================
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],  # 添加早停回调
)

print("Starting training...")
trainer.train()

# 保存最终模型
trainer.save_model(f"./final_model_{SIMPLIFY_RATIO}")