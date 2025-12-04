from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ast_simplifier import JavaCodeSimplifier

# 加载刚才训练好的模型
model_path = "./final_model_0.3"
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base") # Tokenizer通常不变
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

simplifier = JavaCodeSimplifier()

def generate_summary(original_code):
    # 1. 简化
    simplified_code = simplifier.simplify(original_code, remove_ratio=0.3)
    
    print(f"Original Length: {len(original_code.split())}")
    print(f"Simplified Length: {len(simplified_code.split())}")
    
    # 2. 编码
    inputs = tokenizer(
        simplified_code, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    )
    
    # 3. 生成
    summary_ids = model.generate(
        inputs.input_ids, 
        max_length=128, 
        num_beams=4, # Beam Search 效果更好
        early_stopping=True
    )
    
    # 4. 解码
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 测试用例
java_code = """
public int calculateSum(int a, int b) {
    // This function returns the sum of two integers
    int result = a + b;
    return result;
}
"""

print("-" * 30)
print("Generated Summary:", generate_summary(java_code))
print("-" * 30)