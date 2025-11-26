from datasets import load_dataset
from ast_simplifier import JavaCodeSimplifier

# 初始化简化器
simplifier = JavaCodeSimplifier()
REMOVE_RATIO = 0.3  # 设定简化比例

def preprocess_function(examples):
    """
    这个函数会被 map 应用到整个数据集
    """
    inputs = []
    targets = []
    
    for code, doc in zip(examples['code'], examples['docstring']):
        # 1. 应用简化算法
        simplified_code = simplifier.simplify(code, remove_ratio=REMOVE_RATIO)
        inputs.append(simplified_code)
        targets.append(doc)
    
    return {"simplified_code": inputs, "summary": targets}

def load_and_process_data(file_path):
    # 加载本地 JSONL 数据
    dataset = load_dataset('json', data_files=file_path, split='train')
    
    # 为了演示，只取前 1000 条，实际跑的时候去掉这一行
    dataset = dataset.select(range(1000))
    
    # 应用处理
    processed_dataset = dataset.map(preprocess_function, batched=True)
    return processed_dataset

if __name__ == "__main__":
    # 测试一下
    ds = load_and_process_data("java/train.jsonl")
    print(ds[0])