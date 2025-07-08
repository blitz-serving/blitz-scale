import os
import safetensors.torch
import torch

def print_safetensors_info(directory):
    """
    读取指定目录下所有 safetensors 文件，打印名称和形状。

    Args:
        directory (str): 包含 safetensors 文件的目录。
    """

    for filename in os.listdir(directory):
        if filename.endswith(".safetensors"):
            filepath = os.path.join(directory, filename)
            try:
                tensors = safetensors.torch.load_file(filepath)
                print(f"File: {filename}")
                for name, tensor in tensors.items():
                    print(f"  - Name: {name}, Shape: {tensor.shape}, Dtype: {tensor.dtype}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")


if __name__ == "__main__":
    # 替换成包含 safetensors 文件的目录
    # model_directory = "/nvme/huggingface/models/Llama-2-7b-chat-hf"  # 例如: "./models"
    # model_directory = "/nvme/huggingface/models/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
    # model_directory = "/nvme/huggingface/models/Meta-Llama-3-8B"  # 例如: "./models"
    model_directory = "/nvme/huggingface/models/Mistral-Small-24B-Instruct-2501/"  # 例如: "./models"

    print_safetensors_info(model_directory)
