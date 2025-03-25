import os
import sentencepiece as spm
from transformers import LlamaTokenizer

def test_model_path(model_path):
    """检查模型路径是否有效，并确保文件存在"""
    print(f"Checking model path: {model_path}")
    if os.path.exists(model_path):
        print("Model path exists!")
    else:
        print(f"Model path does not exist: {model_path}")
        return False
    return True

def test_model_file_type(model_path):
    """检查文件类型是否正确"""
    if model_path.endswith('.model'):
        print("Valid model file.")
        return True
    else:
        print("Invalid file type. Expected a .model file.")
        return False

def test_sentencepiece_loading(model_path):
    """尝试用 SentencePiece 直接加载模型"""
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        print("SentencePiece model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model with SentencePiece: {e}")
        return False

def test_llama_tokenizer_loading(model_path):
    """尝试加载 LlamaTokenizer"""
    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
        print("LlamaTokenizer loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return False

def main():
    model_path = "path/to/your/llama/model"  # 替换为你的实际模型路径

    # 测试步骤
    if test_model_path(model_path) and test_model_file_type(model_path):
        if test_sentencepiece_loading(model_path) and test_llama_tokenizer_loading(model_path):
            print("Model is ready for use!")
        else:
            print("There was an issue loading the tokenizer.")
    else:
        print("Model path or file type is incorrect.")

if __name__ == "__main__":
    main()
