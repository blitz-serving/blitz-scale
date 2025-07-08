#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../kernel/embedding.h"

namespace blitz::layer {

template<typename DType, typename IdType>
class EmbedLayer {
  private:
    DType* workspace;
    DType* output = nullptr;
    DType* embed_table = nullptr;
    const uint32_t vocab_size;
    const uint32_t hidden_size;
    const uint32_t tp_size;

  public:
    EmbedLayer(uint32_t vocab_, uint32_t hidden_, DType* workspace_ptr, DType** seg_iter, uint32_t tp_size_) :
        workspace(workspace_ptr),
        embed_table(*seg_iter),
        vocab_size(vocab_),
        hidden_size(hidden_),
        tp_size(tp_size_) {
        *seg_iter += (size_t)vocab_size * hidden_size / tp_size;
    }

    ~EmbedLayer() {}

    size_t size_in_bytes() const noexcept {
        return (size_t)vocab_size * (size_t)hidden_size * sizeof(DType) / tp_size;
    }

    void forward(const uint32_t num_tokens, uint32_t* ids, DType* buf = nullptr) {
        output = buf ? buf : workspace;
        blitz::kernel::embedding(output, ids, embed_table, num_tokens, hidden_size);
    }

    void forward(const uint32_t num_tokens, uint32_t* ids, IdType* locs, DType* buf = nullptr) {
        output = buf ? buf : workspace;
        blitz::kernel::embedding(output, ids, locs, embed_table, num_tokens, hidden_size);
    }
};

}  // namespace blitz::layer