import torch
import torch.nn as nn
from transformers import AutoModel, T5EncoderModel

class CodeSearchModel(nn.Module):
    def __init__(self, model_name_or_path, model_type="codebert"):
        super().__init__()
        self.model_type = model_type
        
        # 根据类型加载不同的骨干网络
        if "t5" in model_name_or_path or model_type == "codet5":
            # CodeT5 我们只用 Encoder
            self.encoder = T5EncoderModel.from_pretrained(model_name_or_path)
            self.hidden_size = self.encoder.config.d_model
        else:
            # CodeBERT (RoBERTa)
            self.encoder = AutoModel.from_pretrained(model_name_or_path)
            self.hidden_size = self.encoder.config.hidden_size

        self.loss_fct = nn.CrossEntropyLoss()

    def get_embeddings(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # 统一取第一个 token (CLS 或 T5 的首 token) 作为句子向量
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, code_inputs, query_inputs):
        # 1. 生成向量
        code_vecs = self.get_embeddings(code_inputs["input_ids"], code_inputs["attention_mask"])
        query_vecs = self.get_embeddings(query_inputs["input_ids"], query_inputs["attention_mask"])

        # 2. 计算相似度矩阵 (Batch x Batch)
        scores = torch.matmul(query_vecs, code_vecs.T)
        
        # 3. 对比学习 Loss
        batch_size = scores.size(0)
        labels = torch.arange(batch_size).to(scores.device)
        loss = self.loss_fct(scores, labels)
        
        return loss