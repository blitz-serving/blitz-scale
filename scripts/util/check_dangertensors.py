import safetensors.torch as safe_torch
import torch

'''
model.layers.7.mlp.down_proj.weight: F16 [4096, 11008]
model.layers.7.mlp.gate_proj.weight: F16 [11008, 4096]
model.layers.7.mlp.up_proj.weight: F16 [11008, 4096]
'''

def read_tensor_from_safetensor(filename, tensor_name):
  """
  读取 safetensors 文件中的指定 tensor.

  Args:
    filename: safetensors 文件的路径.
    tensor_name: 要读取的 tensor 的名称.

  Returns:
    如果找到 tensor，则返回 torch.Tensor 对象; 否则返回 None.
  """
  try:
    tensors = safe_torch.load_file(filename)
    if tensor_name in tensors:
      return tensors[tensor_name]
    else:
      print(f"Error: Tensor '{tensor_name}' not found in '{filename}'.")
      return None
  except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    return None
  except Exception as e:
    print(f"Error: An error occurred while reading '{filename}': {e}")
    return None


if __name__ == '__main__':
  #  示例用法:  请替换为你的实际文件名和 tensor 名称
  safetensor_file = "/nvme/huggingface/models/Llama-2-7b-chat-hf/model-00001-of-00002.safetensors"  # 替换为你的文件名
  target_tensor = "model.layers.7.mlp.up_proj.weight"  # 替换为你的 tensor 名称

  tensor_data = read_tensor_from_safetensor(safetensor_file, target_tensor)

  if tensor_data is not None:
    print(f"Tensor '{target_tensor}' loaded successfully.")
    print(f"Tensor shape: {tensor_data.shape}")
    print(f"Tensor dtype: {tensor_data.dtype}")
    # 打印 tensor 的前几个元素 (可选，如果 tensor 很大，不要打印全部)
    # print(f"Tensor values (first 10 elements): {tensor_data.flatten()[:10]}")

    assert tensor_data.is_contiguous()
    print(f"Tensor dtype: {tensor_data.stride()}")
    # print(f"Tensor dtype: {tensor_data[-4:, 0]}")
    # print(f"Tensor dtype: {tensor_data[11007, 0]}")
    target_slice = tensor_data[0:2, 0]
    target_np = target_slice.numpy()
    target_bytes = target_np.tobytes()
    hex_representation = ' '.join([f'0x{byte:02x}' for byte in target_bytes])
    print(hex_representation)
    
    target_slice = tensor_data[0, 0:2]
    target_np = target_slice.numpy()
    target_bytes = target_np.tobytes()
    hex_representation = ' '.join([f'0x{byte:02x}' for byte in target_bytes])
    print(hex_representation)
  else:
    print(f"Failed to load tensor '{target_tensor}'.")