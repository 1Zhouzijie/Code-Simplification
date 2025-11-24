import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import evaluate
import numpy as np
from ast_simplifier import JavaCodeSimplifier
import math
# ================= 配置 =================
MODEL_CHECKPOINT = "Salesforce/codet5-base"
MAX_INPUT_LENGTH = 512  # CodeT5 输入限制
MAX_TARGET_LENGTH = 128 # 摘要输出限制
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
SIMPLIFY_RATIO = 0.3 # 30% 简化率

# ================= 数据准备 =================
# 加载你的数据集 (假设你有 train.jsonl 和 test.jsonl)
data_files = {"train": "java/train.jsonl", "test": "java/test.jsonl"}
dataset = load_dataset("json", data_files=data_files)

# 初始化组件
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
simplifier = JavaCodeSimplifier()
bleu = evaluate.load("sacrebleu")


def preprocess_data(examples):
    # 1. 统一转成列表
    codes = examples['code']
    docstrings = examples['docstring']

    if not isinstance(codes, list):
        codes = [codes]
    if not isinstance(docstrings, list):
        docstrings = [docstrings]

    # 2. 统一转成字符串,处理 None/NaN
    def safe_str(x):
        if x is None:
            return ""
        if isinstance(x, float) and math.isnan(x):
            return ""
        return str(x)

    codes = [safe_str(c) for c in codes]
    docstrings = [safe_str(d) for d in docstrings]

    # 3. 简化代码
    simplified_codes = []
    for c in codes:
        try:
            result = simplifier.simplify(c, SIMPLIFY_RATIO)
            # 强制转成字符串
            simplified_codes.append(str(result) if result is not None else "")
        except:
            simplified_codes.append("")

    # 4. 用占位符替换无效样本,并最终强制类型检查
    final_codes = []
    final_docs = []
    for c, d in zip(simplified_codes, docstrings):
        # 强制转成字符串并去空格
        code_str = str(c).strip() if c else ""
        doc_str = str(d).strip() if d else ""

        # 用占位符替换空字符串
        final_codes.append(code_str if code_str else "empty code")
        final_docs.append(doc_str if doc_str else "No documentation")

    # 5. 最终类型检查:确保每个元素都是字符串
    final_codes = [str(x) for x in final_codes]
    final_docs = [str(x) for x in final_docs]

    # 6. 确保长度一致
    assert len(final_codes) == len(codes)
    assert len(final_docs) == len(docstrings)

    # 7. Tokenize
    model_inputs = tokenizer(
        final_codes,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        final_docs,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 处理数据集
tokenized_datasets = dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=dataset['train'].column_names,
)

# ================= 模型设置 =================
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

# 定义评估指标计算函数
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
        
    # 解码生成的 ID 为文本
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # 替换掉 labels 里的 -100 (PyTorch 忽略的索引)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 后处理：去掉多余空格
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels] # BLEU 需要 list of list

    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

# ================= 训练参数 =================
args = Seq2SeqTrainingArguments(
    output_dir=f"./codet5-simplified-{SIMPLIFY_RATIO}",
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=EPOCHS,
    predict_with_generate=True, # 评估时生成文本
    fp16=torch.cuda.is_available(), # 如果有 GPU 开启混合精度
    push_to_hub=False,
)

# 数据整理器
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ================= 开始训练 =================
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()

# 保存最终模型
trainer.save_model(f"./final_model_{SIMPLIFY_RATIO}")