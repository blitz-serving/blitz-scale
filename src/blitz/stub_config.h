#pragma once

#include <mpi.h>

#include <cstdint>
#include <string>

#include "include/types.hpp"

namespace blitz {

/// \brief Configuration struct for Stub construction, replacing 25+ constructor
///        parameters with a single structured config object.
struct StubConfig {
    bool is_active;
    int world_size;
    rank_t rank;
    int device;
    const char* ib_hca_name;

    // model config
    uint32_t num_layers;
    uint32_t num_qo_head;
    uint32_t num_kv_head;
    uint32_t head_size;
    uint32_t hidden_size;
    uint32_t inter_size;
    uint32_t vocab_size;

    // runtime config
    uint32_t max_batch_size;
    uint32_t max_batch_tokens;

    // kv_cache config
    uint32_t page_size;
    uint32_t max_num_page;

    // communicators
    MPI_Comm machine_comm;
    MPI_Comm tp_comm;

    // parallelism config
    int tp_size;
    int tp_rank;
    int pp_size;

    // model loading
    std::string model_path;
    std::string model_name;
};

}  // namespace blitz
