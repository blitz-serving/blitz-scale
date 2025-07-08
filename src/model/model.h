#pragma once

#include <cstdint>

#include "../kernel/page.h"

namespace blitz {
template<typename DType, typename IdType>
class Stub;
}

namespace blitz::model {
template<typename DType, typename IdType>
class Model {
    friend class Stub<DType, IdType>;

  public:
    Model() = default;

    virtual ~Model() = default;

    virtual void forward(
        const uint32_t num_tokens,
        IdType* ragged_indptr_h,
        blitz::kernel::PageTable<DType, IdType>& page_table,
        const size_t max_num_pages
    ) = 0;

    virtual uint32_t* get_io_token_ids_buf() noexcept = 0;

    virtual uint32_t get_max_position_embeddings() const noexcept = 0;
};
}  // namespace blitz::model