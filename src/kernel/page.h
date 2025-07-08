#pragma once

#include <cstddef>
#include <cstdint>

#include "cuda_helper.h"

namespace blitz::kernel {

/// @note class must be complete
/// Per-batch page table
template<typename DType, typename IdType>
struct PageTable {
    // user manipulate data
    std::vector<IdType> indices;
    std::vector<IdType> indptr;
    std::vector<IdType> last_page_len;
    uint32_t batch_size;

    const uint32_t page_size;
    const uint32_t num_heads;
    const uint32_t head_size;

    // non-user manipulate data
    // page size aka. memory stride of a page
    // 2 * num_heads * page_size * head_size;
    const size_t stride_page;
    // we only support NHD
    // stride n = num_heads * head_size
    const size_t stride_n;
    // stride h = head_size
    const size_t stride_h;

    // pointers to pinned memory
    IdType* indices_h;
    IdType* indptr_h;
    IdType* last_page_len_h;

    /// @note stripped KVCache
    ///     one K page; one V page
    ///     v_data = k_data + num_heads * page_size * head_dim
    DType* k_data;
    DType* v_data;
    /// page index array
    /// compact \sum{seq_len}
    IdType* indices_d;
    /// page index ptr -> page index array
    /// [batch_size + 1]
    /// [0, ..., len(index_list)]
    IdType* indptr_d;
    /// last page token len of each
    IdType* last_page_len_d;

    PageTable(uint32_t batch_size, uint32_t page_size, uint32_t num_heads, uint32_t head_size, DType* base_ptr) :
        batch_size(batch_size),
        page_size(page_size),
        num_heads(num_heads),
        head_size(head_size),
        stride_page(2 * num_heads * page_size * head_size),
        stride_n(num_heads * head_size),
        stride_h(head_size),
        k_data(base_ptr),
        v_data(base_ptr + num_heads * page_size * head_size) {}
};

template<typename DType, typename IdType>
void store_prefill_kv_cache(PageTable<DType, IdType>& page_table, DType* k_data, DType* v_data, IdType* ragged_indptr);

template<typename DType, typename IdType>
void append_decode_kv_cache(PageTable<DType, IdType>& page_table, DType* k_data, DType* v_data);

template<typename DType, typename IdType>
uint32_t cal_total_num_rows(const PageTable<DType, IdType>& page_table);

}  // namespace blitz::kernel
