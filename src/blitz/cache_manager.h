#pragma once

#include <fmt/format.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../kernel/page.h"
#include "include/logger.hpp"

namespace blitz {

template<typename DType, typename IdType>
class CacheManager {
    static constexpr size_t pinned_memory_chunk_size_in_bytes = 2 * 1024 * 1024;
  private:
    // config scratchpad
    const uint32_t local_num_kv_heads;
    const uint32_t local_num_layers;
    const uint32_t &head_size;
    const uint32_t &max_num_page;
    const uint32_t &page_size;
    size_t num_allocated_blocks;
    // mask
    uint8_t* _free_mask;
    std::vector<IdType> _get_free_blocks(size_t num_blocks) const;
    // pinned memory
    IdType* pinned_memory;
    uint8_t* _free_mask_pinned_memory;
    // count pinned usage
    size_t num_allocated_batch;
    int _get_free_batch_meta() const;
    // actual kvcache data
    DType *d_k_cache, *d_v_cache;

  public:
    CacheManager(
        const uint32_t local_num_kv_heads,
        const uint32_t local_num_layers,
        const uint32_t &head_dim,
        const uint32_t &block_size,
        const uint32_t &max_num_pages
    );

    ~CacheManager();

    void allocate(
        const std::vector<std::tuple<size_t, size_t>>& needed_blocks_slots,
        size_t num_blocks,
        std::vector<IdType>& indices,
        std::vector<std::vector<IdType>>& indices_2d,
        const std::vector<IdType>& indptr,
        const std::vector<IdType>& last_page_len,
        IdType** indices_h,
        IdType** indptr_h,
        IdType** last_page_len_h,
        int* pinned_memory_idx
    );
    void append(
        std::vector<IdType>& indices,
        std::vector<std::vector<IdType>>& indices_2d,
        std::vector<IdType>& indptr,
        std::vector<IdType>& last_page_len,
        IdType** indices_h,
        IdType** indptr_h,
        IdType** last_page_len_h,
        const int pinned_memory_idx
    );
    void free(const std::vector<IdType>& block_indices);
    void free_block_meta(int block_meta_idx);

    // helper function
    inline size_t get_free_block_num() const noexcept {
        return max_num_page - num_allocated_blocks;
    }

    inline size_t get_allocated_block_num() const noexcept {
        return num_allocated_blocks;
    }

    inline size_t get_total_block_num() const noexcept {
        return max_num_page;
    }

    inline DType* get_base_ptr() const {
        return d_k_cache;
    }

    inline std::tuple<void*, size_t> get_kvcache_segment() {
        return {
            d_k_cache,
            max_num_page * 2 * local_num_layers
                * (size_t)(local_num_kv_heads * page_size * head_size)
                * sizeof(DType)
        };
    }

    kernel::PageTable<DType, IdType> empty_page_table(uint32_t batch_size
    ) const noexcept {
        return kernel::PageTable<DType, IdType> {
            batch_size,
            page_size,
            local_num_kv_heads,
            head_size,
            d_k_cache
        };
    }
};

}  // namespace blitz

/*---------------------IMPLEMENTATION------------------*/

namespace blitz {

template<typename DType, typename IdType>
CacheManager<DType, IdType>::CacheManager(
    const uint32_t local_num_kv_heads,
    const uint32_t local_num_layers,
    const uint32_t& head_dim,
    const uint32_t& block_size,
    const uint32_t& max_num_pages
) :
    local_num_kv_heads(local_num_kv_heads),
    local_num_layers(local_num_layers),
    head_size(head_dim),
    max_num_page(max_num_pages),
    page_size(block_size),
    num_allocated_blocks(0) {
    size_t one_block_size = local_num_kv_heads * page_size * head_size;
    cudaMalloc(
        &d_k_cache,
        (size_t)sizeof(DType) * max_num_pages * one_block_size * 2
            * local_num_layers
    );
    d_v_cache = d_k_cache + one_block_size;

    this->_free_mask = new uint8_t[max_num_pages];
    memset(_free_mask, 1, max_num_pages);

    // allocate 8MB memory for pinned memory
    cudaMallocHost(&pinned_memory, 256 * pinned_memory_chunk_size_in_bytes);
    this->_free_mask_pinned_memory = new uint8_t[256];
    memset(_free_mask_pinned_memory, 1, 256);
}

template<typename DType, typename IdType>
CacheManager<DType, IdType>::~CacheManager() {
    cudaFree(d_k_cache);
    cudaFreeHost(pinned_memory);
    delete[] this->_free_mask;
    delete[] this->_free_mask_pinned_memory;
}

template<typename DType, typename IdType>
std::vector<IdType>
CacheManager<DType, IdType>::_get_free_blocks(size_t num_blocks) const {
    std::vector<IdType> res;
    res.reserve(num_blocks);
    size_t cnt = 0;
    for (size_t i = 0; i < max_num_page && cnt < num_blocks; ++i) {
        if (_free_mask[i]) {
            cnt++;
            res.push_back(i);
        }
    }
    //TODO throw an exception?
    if (cnt < num_blocks) {
        auto error_message = ::fmt::format(
            "Try to allocate {} blocks, but {} is free!",
            num_blocks,
            cnt
        );
        BZ_ERROR(error_message.c_str());
        throw std::runtime_error(error_message);
    }
    return res;
}

template<typename DType, typename IdType>
int CacheManager<DType, IdType>::_get_free_batch_meta() const {
    for (size_t i = 0; i < 256; ++i) {
        if (_free_mask_pinned_memory[i]) {
            _free_mask_pinned_memory[i] = 0;
            return i;
        }
    }
    return -1;
}

template<typename DType, typename IdType>
void CacheManager<DType, IdType>::allocate(
    const std::vector<std::tuple<size_t, size_t>>& needed_blocks_slots,
    size_t num_blocks,
    std::vector<IdType>& indices,
    std::vector<std::vector<IdType>>& indices_2d,
    const std::vector<IdType>& indptr,  // init outside
    const std::vector<IdType>& last_page_len,  // init outside
    IdType** h_indices,
    IdType** h_indptr,
    IdType** h_last_page_len,
    int* pinned_memory_idx
) {
    auto free_block_indices = _get_free_blocks(num_blocks);
    auto free_batch_meta_idx = _get_free_batch_meta();
    assert(free_batch_meta_idx >= 0);
    *pinned_memory_idx = free_batch_meta_idx;
    assert((needed_blocks_slots.size() + 1 == indptr.size()));
    assert((needed_blocks_slots.size() == last_page_len.size()));
    auto free_block_iter = free_block_indices.begin();

    IdType* cummulated_pinned_memory_pos = pinned_memory
        + free_batch_meta_idx * pinned_memory_chunk_size_in_bytes
            / sizeof(IdType);
    for (auto iter : needed_blocks_slots) {
        auto [need_blocks, need_slots] = iter;
        std::vector<IdType> blocks = {
            free_block_iter,
            free_block_iter + need_blocks
        };
        indices.insert(indices.end(), blocks.begin(), blocks.end());
        for (auto block : blocks) {
            _free_mask[block] = 0;
        }
        indices_2d.emplace_back(std::move(blocks));
        free_block_iter += need_blocks;
        num_allocated_blocks += need_blocks;
    }
    // page_indices
    std::copy(indices.begin(), indices.end(), cummulated_pinned_memory_pos);
    *h_indices = cummulated_pinned_memory_pos;
    cummulated_pinned_memory_pos += indices.size();

    // indptr
    std::copy(indptr.begin(), indptr.end(), cummulated_pinned_memory_pos);
    *h_indptr = cummulated_pinned_memory_pos;
    cummulated_pinned_memory_pos += indptr.size();

    // last_page_len
    std::copy(
        last_page_len.begin(),
        last_page_len.end(),
        cummulated_pinned_memory_pos
    );
    *h_last_page_len = cummulated_pinned_memory_pos;
    num_allocated_batch += 1;
}

template<typename DType, typename IdType>
void CacheManager<DType, IdType>::append(
    std::vector<IdType>& indices,
    std::vector<std::vector<IdType>>& indices_2d,
    std::vector<IdType>& indptr,
    std::vector<IdType>& last_page_len,
    IdType** indices_h,
    IdType** indptr_h,
    IdType** last_page_len_h,
    const int pinned_memory_id
) {
    size_t batch_size = indices_2d.size();
    size_t num_old_blocks = indices.size();
    assert((last_page_len.size() == batch_size));

    std::vector<size_t> full_idx;
    for (size_t i = 0; i < batch_size; ++i) {
        if (last_page_len[i] == (IdType)this->page_size) {
            full_idx.push_back(i);
        } else {
            last_page_len[i] += 1;
        }
    }
    if (full_idx.empty()) {
        std::copy(last_page_len.begin(), last_page_len.end(), *last_page_len_h);
        return;
    }

    // if no exception, then allocation is success
    auto new_blocks = _get_free_blocks(full_idx.size());
    for (size_t i = 0; i < full_idx.size(); ++i) {
        size_t seq_idx = full_idx[i];
        indices_2d[seq_idx].push_back(new_blocks[i]);
        _free_mask[new_blocks[i]] = 0;
        last_page_len[seq_idx] = 1;
    }
    indices.erase(indices.begin() + indptr[full_idx[0]], indices.end());
    for (size_t i = full_idx[0]; i < batch_size; ++i) {
        indices
            .insert(indices.end(), indices_2d[i].begin(), indices_2d[i].end());
        indptr[i + 1] = indptr[i] + indices_2d[i].size();
    }
    // BZ_DEBUG(
    //     "num_old_blocks={}, new_blocks={}, indices={}",
    //     num_old_blocks, new_blocks.size(), indices.size()
    // );
    assert(((num_old_blocks + new_blocks.size()) == indices.size()));

    // flush pinned memory region
    assert(
        (*indices_h
         == pinned_memory
             + pinned_memory_id * pinned_memory_chunk_size_in_bytes
                 / sizeof(IdType))
    );
    IdType* pinned_memory_base = *indices_h;
    std::copy(indices.begin(), indices.end(), pinned_memory_base);
    pinned_memory_base += indices.size();
    std::copy(indptr.begin(), indptr.end(), pinned_memory_base);
    *indptr_h = pinned_memory_base;
    pinned_memory_base += indptr.size();
    std::copy(last_page_len.begin(), last_page_len.end(), pinned_memory_base);
    *last_page_len_h = pinned_memory_base;
    pinned_memory_base += last_page_len.size();

    assert(
        ((pinned_memory_base - *indices_h)
         <= (long)(pinned_memory_chunk_size_in_bytes / sizeof(IdType)))
    );
}

template<typename DType, typename IdType>
void CacheManager<DType, IdType>::free(const std::vector<IdType>& block_indices
) {
    for (auto idx : block_indices)
        this->_free_mask[idx] = 1;
    num_allocated_blocks -= block_indices.size();
}

template<typename DType, typename IdType>
void CacheManager<DType, IdType>::free_block_meta(int block_meta_idx) {
    this->_free_mask_pinned_memory[block_meta_idx] = 1;
    num_allocated_batch -= 1;
}

}  // namespace blitz