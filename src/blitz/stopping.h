
#pragma once

#include "batch.h"
#include "include/logger.hpp"

namespace blitz {

template <typename DType, typename IdType>
struct Batch;

namespace StoppingCriteria {

// template <typename DType, typename IdType>
// bool eos_token(const Batch<DType, IdType> &batch, size_t i) {
//     constexpr int64_t END_TOKEN = 50118; // TODO WHY NOT 2?
//     return batch.all_input_tokens[i].back() != END_TOKEN;
// }

template <typename DType, typename IdType>
bool length(const Batch<DType, IdType> &batch, size_t i) {
    return batch.all_tokens[i].size() < batch.num_total_tokens[i];
}

} // namespace StoppingCriteria

} // namespace blitz
