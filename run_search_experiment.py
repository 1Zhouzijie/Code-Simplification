import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gc
import json
from ast_simplifier import JavaCodeSimplifier
from model_search import CodeSearchModel

# ================= 1. 全局配置 (在此修改) =================
# 本地模型路径 (请修改为您实际的本地路径)
LOCAL_MODEL_PATHS = {
    "codebert": "./codebert-base-local",  # 请确保已下载并解压到此文件夹
    "codet5": "./codet5-base-local"       # 您已有的 CodeT5 路径
}

# 数据文件路径
TRAIN_FILE = "java_research/train.txt"   # 您的 txt 训练文件
TEST_FILE = "java_research/java_test_0.jsonl"   # 您的 jsonl 测试文件

RATIOS_TO_TEST = [0,0.1,0.3,0.5]
MODELS_TO_TEST = [ "codet5"]

BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MAX_LENGTH = 256
NUM_WORKERS = 0 

# 显卡加速配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ================= 2. 数据读取函数 (核心修改) =================
def load_custom_dataset():
    """
    自定义加载逻辑：
    - Train: 读取 .txt 文件
    - Test: 读取 .jsonl 文件
    """
    print(f"Loading datasets...")
    
    # --- 1. 加载训练集 (TXT) ---
    train_codes = []
    train_docs = []
    
    # 假设 txt 文件每一行是用 TAB 分隔的: "code \t docstring"
    # 如果您的格式不同 (比如两行一条)，请告诉我，我再改
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 尝试按 TAB 分割
            parts = line.split('\t')
            if len(parts) >= 2:
                # 假设第一列是 code，第二列是 doc (根据实际情况调整)
                train_codes.append(parts[0]) 
                train_docs.append(parts[1])
            else:
                # 如果没有 TAB，可能这行全是代码，或者格式不对，暂且跳过或作为单列处理
                pass 
    
    print(f"  Loaded {len(train_codes)} training samples from txt.")
    train_dataset = Dataset.from_dict({"code": train_codes, "docstring": train_docs})

    # --- 2. 加载测试集 (JSONL) ---
    test_codes = []
    test_docs = []
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
                # 兼容不同的 key 名，比如有的数据集叫 'code_tokens' 有的叫 'code'
                c = item.get('code') or item.get('function_tokens') or ""
                d = item.get('docstring') or item.get('docstring_tokens') or ""
                
                # 如果是 list (tokenized)，转回 string
                if isinstance(c, list): c = " ".join(c)
                if isinstance(d, list): d = " ".join(d)
                
                test_codes.append(c)
                test_docs.append(d)
            except:
                continue
                
    print(f"  Loaded {len(test_codes)} test samples from jsonl.")
    test_dataset = Dataset.from_dict({"code": test_codes, "docstring": test_docs})
    
    return train_dataset, test_dataset


# ================= 3. 辅助评估与训练逻辑 =================
def evaluate_mrr(model, dataloader):
    """计算 MRR 指标"""
    model.eval()
    code_vecs = []
    query_vecs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
            code_inputs, query_inputs = batch
            code_inputs = {k: v.to(DEVICE) for k, v in code_inputs.items()}
            query_inputs = {k: v.to(DEVICE) for k, v in query_inputs.items()}
            
            c_vec = model.get_embeddings(code_inputs['input_ids'], code_inputs['attention_mask'])
            q_vec = model.get_embeddings(query_inputs['input_ids'], query_inputs['attention_mask'])
            
            code_vecs.append(c_vec.cpu().numpy())
            query_vecs.append(q_vec.cpu().numpy())
            
    code_vecs = np.concatenate(code_vecs, 0)
    query_vecs = np.concatenate(query_vecs, 0)
    
    scores = np.matmul(query_vecs, code_vecs.T)
    
    ranks = []
    for i in range(len(scores)):
        score_row = scores[i]
        sorted_indices = np.argsort(-score_row)
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(1.0 / rank)
        
    return np.mean(ranks)


def run_single_experiment(model_type, ratio, train_ds_raw, test_ds_raw):
    """运行单个实验"""
    print(f"\n{'-'*60}")
    print(f"🚀 Experiment: Model={model_type}, Ratio={ratio}")
    print(f"{'-'*60}")
    
    model_path = LOCAL_MODEL_PATHS[model_type]
    # 如果本地没有，回退到在线加载
    if not os.path.exists(model_path):
        print(f"Warning: Local path {model_path} not found. Downloading from HuggingFace...")
        model_path = "microsoft/codebert-base" if model_type == "codebert" else "Salesforce/codet5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    simplifier = JavaCodeSimplifier()
    
    # --- 数据处理 ---
    def preprocess_batch(batch):
        codes = batch['code']
        docstrings = batch['docstring']
        simplified_codes = []
        for c in codes:
            try:
                s_code = simplifier.simplify(c, remove_ratio=ratio)
                simplified_codes.append(str(s_code))
            except:
                simplified_codes.append("")
        
        cleaned_docs = []
        for d in docstrings:
            d = str(d) if d is not None else ""
            cleaned_docs.append(d.split('\n')[0]) 
            
        return simplified_codes, cleaned_docs

    def collate_fn(batch):
        code_list = [item['code_simplified'] for item in batch]
        doc_list = [item['doc_cleaned'] for item in batch]
        
        code_inputs = tokenizer(code_list, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        query_inputs = tokenizer(doc_list, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return code_inputs, query_inputs

    def map_function(examples):
        c, d = preprocess_batch(examples)
        return {"code_simplified": c, "doc_cleaned": d}

    # 处理数据集
    train_ds = train_ds_raw.map(map_function, batched=True, batch_size=100, load_from_cache_file=False, desc="Processing Train")
    test_ds = test_ds_raw.map(map_function, batched=True, batch_size=100, load_from_cache_file=False, desc="Processing Test")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    
    # --- 模型训练 ---
    model = CodeSearchModel(model_path, model_type=model_type)
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    best_mrr = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in pbar:
            code_inputs, query_inputs = batch
            code_inputs = {k: v.to(DEVICE) for k, v in code_inputs.items()}
            query_inputs = {k: v.to(DEVICE) for k, v in query_inputs.items()}
            
            optimizer.zero_grad()
            loss = model(code_inputs, query_inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        mrr = evaluate_mrr(model, test_loader)
        print(f"  Epoch {epoch+1} finished. MRR: {mrr:.4f}")
        if mrr > best_mrr: best_mrr = mrr
            
    # 清理
    del model, optimizer, train_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_mrr

# ================= 4. 主程序 =================
if __name__ == "__main__":
    # 1. 先加载原始数据 (只加载一次，不用每次循环都读文件)
    raw_train_ds, raw_test_ds = load_custom_dataset()
    

    raw_train_ds = raw_train_ds.select(range(1000))
    raw_test_ds = raw_test_ds.select(range(100))

    results = {m: [] for m in MODELS_TO_TEST}
    
    try:
        for model_type in MODELS_TO_TEST:
            for ratio in RATIOS_TO_TEST:
                score = run_single_experiment(model_type, ratio, raw_train_ds, raw_test_ds)
                results[model_type].append(score)
    except KeyboardInterrupt:
        print("\nInterrupted.")

    # 绘图逻辑
    print("\nResults:", results)
    if any(len(v) > 0 for v in results.values()):
        plt.figure(figsize=(10, 6))
        for m, scores in results.items():
            if scores:
                plt.plot(RATIOS_TO_TEST[:len(scores)], scores, marker='o', label=m)
        plt.title('Code Search MRR vs Simplification Ratio')
        plt.xlabel('Simplification Ratio')
        plt.ylabel('MRR')
        plt.legend()
        plt.grid(True)
        plt.savefig('search_benchmark_result.png')
        print("Chart saved.")