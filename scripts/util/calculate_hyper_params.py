KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024

HYPERPARAM_LLAMA2_7B = {
    "vocab_size": 32000,
    "max_position_embeddings": 4096,
    "hidden_size": 4096,
    "num_layers": 32,
    "num_q_heads": 32,
    "num_kv_heads": 32,
    "head_dim": 128,
    "ffn_inter_dim": 11008,
    "params_size": 14 * GB,
}

HYPERPARAM_LLAMA3_8B = {
    "vocab_size": 128256,
    "max_position_embeddings": 4096,
    "hidden_size": 4096,
    "num_layers": 32,
    "num_q_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "ffn_inter_dim": 14336,
    "params_size": 16 * GB,
}

HYPERPARAM_LLAMA2_13B = {
    "vocab_size": 32000,
    "max_position_embeddings": 4096,
    "hidden_size": 5120,
    "num_layers": 40,
    "num_q_heads": 40,
    "num_kv_heads": 40,
    "head_dim": 128,
    "ffn_inter_dim": 13824,
    "params_size": 26 * GB,
}

HYPERPARAM_MISTRAL_13B = {
    "vocab_size": 32003,
    "max_position_embeddings": 4096,
    "hidden_size": 4096,
    "num_layers": 60,
    "num_q_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "ffn_inter_dim": 14336,
    "params_size": 26 * GB,
}

HYPERPARAMS = {
    "llama2_7b": HYPERPARAM_LLAMA2_7B,
    "llama3_8b": HYPERPARAM_LLAMA3_8B,
    "llama2_13b": HYPERPARAM_LLAMA2_13B,
    "mistral_13b": HYPERPARAM_MISTRAL_13B,
}


def calculate_kv_cache_per_token_per_layer(params):
    return params["head_dim"] * params["num_kv_heads"] * 2 * 2


def calculate_kv_cache_per_token(params):
    return calculate_kv_cache_per_token_per_layer(params) * params["num_layers"]


def calculate_block_size(params):
    return calculate_kv_cache_per_token(params) * 16


if __name__ == "__main__":
    for model_name, params in HYPERPARAMS.items():
        print(f"Model {model_name}")
        print(f"KV cache per token: {calculate_kv_cache_per_token(params) / MB} MB")
        print(f"Block size: {calculate_block_size(params)}")
        print(
            f"Block num {(37 * GB - params['params_size']) / calculate_block_size(params)}"
        )
        print(
            f"Element size: {calculate_kv_cache_per_token_per_layer(params) * 16 / KB} KB"
        )
        print(
            f"Tokens transferred persec {16 * GB / calculate_kv_cache_per_token(params)}"
        )
        print("==============================")
