#pragma once

#include <iostream>
#include <vector>

namespace blitz::kernel {
enum class ActivationType { RELU, SILU, GELU };

}

namespace blitz::model {

struct GptHyperParam {
    // Hyper-parameters
    int64_t vocab_size;  // The size of the vocabulary
    int64_t max_position_embeddings;  // The maximum length of the input sequence
    int64_t hidden_size;  // The length of the embedded vector
    int64_t num_layers;  // The number of layers (transformer blocks)
    int64_t num_q_heads;  // The number of query heads in the multi-head attention
    int64_t num_kv_heads;  // The number of key/value heads in the multi-head attention.
        // If the model does not use GQA (Grouped Query Attention), just
        // set num_kv_heads = num_q_heads
    int64_t head_dim;  // The dimension of each head (length of the key, query, and value vectors)
    int64_t ffn_inter_dim;  // The intermediate dimension of the feed-forward network

    // Model configurations
    bool is_pre_layernorm;  // Perform layernorm before/after the self-attention and feed-forward network
    bool is_rotary_posi_embedding;  // Use rotary position embedding instead of absolute position embedding
    bool is_gated_ffn;  // Use gated feed-forward network
    blitz::kernel::ActivationType ffn_activation_type;  // The activation function of the feed-forward network
    bool is_rmsnorm;  // Use RMSNorm instead of LayerNorm
    bool is_attn_qkv_biased;
    bool is_attn_out_biased;

    friend std::ostream& operator<<(std::ostream& os, const GptHyperParam& params) {
        os << "GptHyperParam {\n"
           << "\tvocab_size = " << params.vocab_size << "\n"
           << "\tmax_position_embeddings = " << params.max_position_embeddings << "\n"
           << "\thidden_size = " << params.hidden_size << "\n"
           << "\tnum_layers = " << params.num_layers << "\n"
           << "\tnum_q_heads = " << params.num_q_heads << "\n"
           << "\tnum_kv_heads = " << params.num_kv_heads << "\n"
           << "\thead_dim = " << params.head_dim << "\n"
           << "\tffn_inter_dim = " << params.ffn_inter_dim << "\n"
           << "\tis_pre_layernorm = " << params.is_pre_layernorm << "\n"
           << "\tis_rotary_posi_embedding = " << params.is_rotary_posi_embedding << "\n"
           << "\tis_gated_ffn = " << params.is_gated_ffn << "\n"
           << "\tffn_activation_type = " << static_cast<int>(params.ffn_activation_type) << "\n"
           << "\tis_rmsnorm = " << params.is_rmsnorm << "\n"
           << "\tis_attn_qkv_biased = " << params.is_attn_qkv_biased << "\n"
           << "\tis_attn_out_bias = " << params.is_attn_out_biased << "\n"
           << "}";
        return os;
    }

    static GptHyperParam GetOptHyperParam(
        int64_t vocab_size,
        int64_t max_position_embeddings,
        int64_t hidden_size,
        int64_t num_layers,
        int64_t num_heads,
        int64_t head_dim,
        int64_t ffn_inter_dim
    ) {
        return GptHyperParam {
            .vocab_size = vocab_size,
            .max_position_embeddings = max_position_embeddings,
            .hidden_size = hidden_size,
            .num_layers = num_layers,
            .num_q_heads = num_heads,
            .num_kv_heads = num_heads,
            .head_dim = head_dim,
            .ffn_inter_dim = ffn_inter_dim,
            .is_pre_layernorm = true,
            .is_rotary_posi_embedding = false,
            .is_gated_ffn = false,
            .ffn_activation_type = blitz::kernel::ActivationType::RELU,
            .is_rmsnorm = false,
            .is_attn_qkv_biased = true,
            .is_attn_out_biased = true
        };
    }

    static GptHyperParam GetLlama2HyperParam(
        int64_t vocab_size,
        int64_t max_position_embeddings,
        int64_t hidden_size,
        int64_t num_layers,
        int64_t num_q_heads,
        int64_t num_kv_heads,
        int64_t head_dim,
        int64_t ffn_inter_dim
    ) {
        return GptHyperParam {
            .vocab_size = vocab_size,
            .max_position_embeddings = max_position_embeddings,
            .hidden_size = hidden_size,
            .num_layers = num_layers,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .ffn_inter_dim = ffn_inter_dim,
            .is_pre_layernorm = true,
            .is_rotary_posi_embedding = true,
            .is_gated_ffn = true,
            .ffn_activation_type = blitz::kernel::ActivationType::SILU,
            .is_rmsnorm = true,
            .is_attn_qkv_biased = false,
            .is_attn_out_biased = false
        };
    }

    static GptHyperParam GetGpt2HyperParam(
        int64_t vocab_size,
        int64_t max_position_embeddings,
        int64_t hidden_size,
        int64_t num_layers,
        int64_t num_heads,
        int64_t head_dim,
        int64_t ffn_inter_dim
    ) {
        return GptHyperParam {
            .vocab_size = vocab_size,
            .max_position_embeddings = max_position_embeddings,
            .hidden_size = hidden_size,
            .num_layers = num_layers,
            .num_q_heads = num_heads,
            .num_kv_heads = num_heads,
            .head_dim = head_dim,
            .ffn_inter_dim = ffn_inter_dim,
            .is_pre_layernorm = true,
            .is_rotary_posi_embedding = false,
            .is_gated_ffn = false,
            .ffn_activation_type = blitz::kernel::ActivationType::GELU,
            .is_rmsnorm = false,
            .is_attn_qkv_biased = true,
            .is_attn_out_biased = true
        };
    }
};

struct GptPagedAttnParam {
    // Hyperparameters related to PagedAttention
    int64_t block_size;
    int64_t max_num_block_per_req;

    friend std::ostream& operator<<(std::ostream& os, const GptPagedAttnParam& params) {
        os << "GptPagedAttnParam {\n"
           << "\tblock_size = " << params.block_size << "\n"
           << "\tmax_num_block_per_req = " << params.max_num_block_per_req << "\n"
           << "}";
        return os;
    }
};

struct GptParallelismParam {
    // Hyper parameters related to parallelism
    int64_t tensor_para_size = 1;
    int64_t tensor_para_rank = 0;

    int64_t pipeline_para_size = 1;
    int64_t pipeline_para_rank = 0;

    bool hyper_inited = false;

    // The following two parameters are used for pipeline parallelism
    // The layer range of the current pipeline stage is [layer_begin, layer_end)
    int64_t layer_begin = 0, layer_end = 0, local_layer_num = 0;

    GptParallelismParam(
        int64_t tensor_para_size = 1,
        int64_t tensor_para_rank = 0,
        int64_t pipeline_para_size = 1,
        int64_t pipeline_para_rank = 0
    ) :
        tensor_para_size(tensor_para_size),
        tensor_para_rank(tensor_para_rank),
        pipeline_para_size(pipeline_para_size),
        pipeline_para_rank(pipeline_para_rank) {}

    GptParallelismParam(const std::vector<int64_t> parallel_config) :
        GptParallelismParam(parallel_config[0], parallel_config[1], parallel_config[2], parallel_config[3]) {}

    void init_by_hyper_param(const GptHyperParam& hyper_param) {
        if (hyper_inited) {
            return;
        }
        hyper_inited = true;
        if (hyper_param.num_layers % pipeline_para_size != 0) {
            throw std::invalid_argument("The number of layers must be divisible by the pipeline parallelism size.");
        }
        local_layer_num = hyper_param.num_layers / pipeline_para_size;
        layer_begin = pipeline_para_rank * local_layer_num;
        layer_end = layer_begin + local_layer_num;
    }

    inline bool is_parallel() const {
        return tensor_para_size > 1 || pipeline_para_size > 1;
    }

    inline bool is_last_stage() const {
        return pipeline_para_rank == pipeline_para_size - 1;
    }

    inline bool is_first_stage() const {
        return pipeline_para_rank == 0;
    }

    inline bool is_stage_leader() const {
        return tensor_para_rank == 0;
    }

    void set_parallelism(
        int64_t tensor_para_size,
        int64_t tensor_para_rank,
        int64_t pipeline_para_size,
        int64_t pipeline_para_rank
    ) {
        this->tensor_para_size = tensor_para_size;
        this->tensor_para_rank = tensor_para_rank;
        this->pipeline_para_size = pipeline_para_size;
        this->pipeline_para_rank = pipeline_para_rank;
    }

    friend std::ostream& operator<<(std::ostream& os, const GptParallelismParam& param) {
        os << "tensor_para_size: " << param.tensor_para_size << std::endl;
        os << "tensor_para_rank: " << param.tensor_para_rank << std::endl;
        os << "pipeline_para_size: " << param.pipeline_para_size << std::endl;
        os << "pipeline_para_rank: " << param.pipeline_para_rank << std::endl;
        return os;
    }
};

}  // namespace blitz::model