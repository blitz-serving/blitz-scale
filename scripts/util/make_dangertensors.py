import os
import safetensors.torch
import torch
import struct
import json
from typing import List


def generate_tensor_order(nlayers):
    tensor_order = [
        "model.embed_tokens.weight",
    ]
    for i in range(nlayers):
        tensor_order_p = [
            "model.layers.{}.input_layernorm.weight",
            "model.layers.{}.self_attn.q_proj.weight",
            "model.layers.{}.self_attn.k_proj.weight",
            "model.layers.{}.self_attn.v_proj.weight",
            "model.layers.{}.self_attn.o_proj.weight",
            "model.layers.{}.post_attention_layernorm.weight",
            "model.layers.{}.mlp.gate_proj.weight",
            "model.layers.{}.mlp.up_proj.weight",
            "model.layers.{}.mlp.down_proj.weight",
        ]
        tensor_order += list(map(lambda p: p.format(i), tensor_order_p))
    tensor_order += [
        "model.norm.weight",
        "lm_head.weight",
    ]
    return tensor_order


def process_and_write_tensors(
    directory, output_path, tensor_order: List[str], tp_size: int
):
    """
    读取指定目录下所有 safetensors 文件，转置所有 2D 张量，并按指定顺序写入二进制文件。

    Args:
        directory (str): 包含 safetensors 文件的目录。
        output_file (str): 输出二进制文件的路径。
        tensor_order (list): 张量名称的顺序，用于写入文件。
    """

    all_tensors = {}
    config_file = os.path.join(directory, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    inter_size = config["intermediate_size"]
    head_dim = (
        config["hidden_size"] // config["num_attention_heads"]
        if not "head_dim" in config.keys()
        else config["head_dim"]
    )
    num_qo_head = config["num_attention_heads"]
    num_kv_head = config["num_key_value_heads"]

    print(f"hidden_size: {hidden_size}, inter_size: {inter_size}, head_dim: {head_dim}")

    for filename in os.listdir(directory):
        if filename.endswith(".safetensors"):
            filepath = os.path.join(directory, filename)
            try:
                tensors = safetensors.torch.load_file(filepath)
                all_tensors.update(tensors)  # 将所有张量添加到字典中
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                return

    # 检查 tensor_order 中是否存在所有张量名称
    for tensor_name in tensor_order:
        if tensor_name not in all_tensors:
            print(f"Error: Tensor '{tensor_name}' not found in safetensors files.")
            return

    for tp_rank in range(tp_size):
        filename = "dangertensors.{}.bin".format(tp_rank)
        output_file = os.path.join(output_path, filename)
        with open(output_file, "wb") as f:
            tensor_order_it = iter(tensor_order)
            for tensor_name in tensor_order_it:
                tensor = all_tensors[tensor_name]
                tensors = []
                print(f"[INIT] Name: {tensor_name}, shape: {tensor.shape}")

                if len(tensor.shape) == 2:
                    # NOTE: fuse gate & up proj
                    if "gate_proj" in tensor_name:
                        gate_tensor = tensor
                        up_tensor = all_tensors[next(tensor_order_it)]
                        assert gate_tensor.shape == up_tensor.shape

                        gate_tensor_slice = gate_tensor[
                            (tp_rank * (inter_size // tp_size)) : (
                                (tp_rank + 1) * (inter_size // tp_size)
                            ),
                            :,
                        ]
                        up_tensor_slice = up_tensor[
                            (tp_rank * (inter_size // tp_size)) : (
                                (tp_rank + 1) * (inter_size // tp_size)
                            ),
                            :,
                        ]
                        result = torch.transpose(
                            torch.cat((gate_tensor_slice, up_tensor_slice), dim=0), 0, 1
                        )
                        result = result.contiguous()
                        tensors.append(result)
                        tensor_name = tensor_name.replace("gate_proj", "gate_up_proj")
                        print("------------------------")
                    # down_proj
                    elif "down_proj" in tensor_name:
                        tensor_slice = tensor[
                            :,
                            (tp_rank * (inter_size // tp_size)) : (
                                (tp_rank + 1) * (inter_size // tp_size)
                            ),
                        ]
                        tensor_slice = torch.transpose(tensor_slice, 0, 1)
                        tensor_slice = tensor_slice.contiguous()
                        tensors.append(tensor_slice)
                    else:
                        if "lm_head" in tensor_name:
                            # [vocab_size, hidden_size]
                            assert (vocab_size, hidden_size) == tensor.shape
                            assert vocab_size % tp_size == 0
                            tensor_slice = tensor[
                                (tp_rank * (vocab_size // tp_size)) : (
                                    (tp_rank + 1) * (vocab_size // tp_size)
                                ),
                                :,
                            ]
                            tensor_slice = torch.transpose(tensor_slice, 0, 1)
                            tensor_slice = tensor_slice.contiguous()
                            tensors.append(tensor_slice)
                        elif "embed" in tensor_name:
                            # [vocab_size, hidden_size]
                            assert (vocab_size, hidden_size) == tensor.shape
                            assert vocab_size % tp_size == 0
                            tensor_slice = tensor[
                                (tp_rank * (vocab_size // tp_size)) : (
                                    (tp_rank + 1) * (vocab_size // tp_size)
                                ),
                                :,
                            ]
                            tensor_slice = tensor_slice.contiguous()
                            tensors.append(tensor_slice)
                        else:
                            assert "self_attn" in tensor_name
                            assert num_qo_head % tp_size == 0
                            assert num_kv_head % tp_size == 0

                            if "o_proj" in tensor_name:  # Mistral-24B
                                # [hidden_size, num_qo_head * head_size]
                                slice_head_num = num_qo_head // tp_size
                                start_head_idx = tp_rank * slice_head_num
                                tensor_slice = tensor[
                                    :,
                                    start_head_idx
                                    * head_dim : (start_head_idx + slice_head_num)
                                    * head_dim,
                                ]
                            elif "q_proj" in tensor_name:
                                # [num_qo_head * head_size, hidden_size]
                                slice_head_num = num_qo_head // tp_size
                                start_head_idx = tp_rank * slice_head_num
                                tensor_slice = tensor[
                                    start_head_idx
                                    * head_dim : (start_head_idx + slice_head_num)
                                    * head_dim,
                                    :,
                                ]
                            else:
                                # [num_kv_head * head_size, hidden_size]
                                slice_head_num = num_kv_head // tp_size
                                start_head_idx = tp_rank * slice_head_num
                                tensor_slice = tensor[
                                    start_head_idx
                                    * head_dim : (start_head_idx + slice_head_num)
                                    * head_dim,
                                    :,
                                ]
                            tensor_slice = torch.transpose(tensor_slice, 0, 1)
                            tensor_slice = tensor_slice.contiguous()
                            tensors.append(tensor_slice)
                        print("========================")
                else:
                    # 1-D tensor
                    tensors.append(tensor)
                    assert len(tensor.shape) == 1
                # 将张量转换为 NumPy 数组，以便更轻松地处理数据类型
                for tensor in tensors:
                    print(
                        f"Name: {tensor_name}, dtype: {tensor.dtype}, shape: {tensor.shape}"
                    )
                    try:
                        uint8_tensor = tensor.view(torch.uint8)
                    except Exception:
                        # print(f"Tensor stride: {tensor.stride()}")
                        if not tensor.is_contiguous():
                            tensor = tensor.contiguous()
                            uint8_tensor = tensor.view(torch.uint8)
                    np_array = uint8_tensor.numpy()
                    bytes_data = np_array.tobytes()
                    f.write(bytes_data)
                    # print(bytes_data)

                # for tensor in tensors:

                print("==========================")

                # 写入张量数据
                # tensor_bytes = tensor_np.tobytes()
                # f.write(tensor_bytes)


if __name__ == "__main__":
    # NOTE README:
    # change `model_name` to model path
    # XXX don't forget to change num_layer (80, 40, 32, etc.)
    # XXX don't forget to change tp_size (1, 2, 4, etc.)
    # model_name = 'Llama-2-7b-chat-hf'
    # model_name = 'DeepSeek-R1-Distill-Llama-8B'
    # model_name = "Mistral-Small-24B-Instruct-2501"
    model_name = 'Llama-3-8B-Instruct'
    model_directory = "/nvme/ly/models/{}".format(model_name)
    output_path = "/nvme/ly/models/{}/".format(model_name)
    tensor_order = generate_tensor_order(32)
    tp_size = 1
    process_and_write_tensors(model_directory, output_path, tensor_order, tp_size)
