from transformers import LlamaTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"  # 确认这个名称是正确的

# 使用访问令牌加载模型
tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=True)
