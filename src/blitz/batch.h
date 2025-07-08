#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../include/tokenizer.hpp"
#include "cache_manager.h"
#include "fmt/format.h"
#include "generate.pb.h"
#include "include/logger.hpp"
#include "kernel/page.h"
#include "stopping.h"

namespace blitz {

/// \brief define types to distinguish overloaded funcitons
namespace batch {

struct prefill_t {
    explicit prefill_t() = default;
};

inline constexpr prefill_t prefill {};

struct decode_t {
    explicit decode_t() = default;
};

inline constexpr decode_t decode {};

}  // namespace batch

template<typename DType, typename IdType>
struct Batch {
    /// \brief KVCache
    CacheManager<DType, IdType>* cache_manager;
    kernel::PageTable<DType, IdType> page_table;
    // per request page indices
    std::vector<std::vector<IdType>> indices_2d;
    // meta data for PageTable host pointer
    int pinned_memory_idx = -1;

    // Ragged Tensor
    std::vector<IdType> ragged_indptr;
    size_t max_tokens;
    uint32_t num_tokens = 0;

    /// \brief Request
    // batch id
    uint64_t id;
    size_t batch_size;
    std::vector<uint64_t> request_ids;
    std::vector<size_t> num_prompt_tokens;
    std::vector<size_t> num_total_tokens;
    std::vector<size_t> num_prompt_blocks;  // use by migration manager

    /// \brief Generation
    std::vector<std::vector<uint32_t>> all_tokens;
    std::function<bool(const Batch&, size_t)> not_stopped;

    Batch() = default;
    Batch(const Batch<DType, IdType>&) = delete;
    Batch<DType, IdType>& operator=(const Batch<DType, IdType>&) = delete;

    // all trivial types; destructor need to do nothing
    ~Batch() {}

    Batch(
        batch::prefill_t,
        size_t batch_max_tokens,
        uint64_t batch_id,
        size_t batch_size,
        std::vector<uint32_t>& input_tokens,
        uint64_t request_id,
        uint32_t max_new_tokens,
        CacheManager<DType, IdType>* cache_manager_,
        const HuggingfaceTokenizer& tokenizer
    );

    Batch(
        batch::prefill_t,
        const generate::v2::Batch& pb_batch,
        CacheManager<DType, IdType>* cache_manager_,
        const HuggingfaceTokenizer& tokenizer
    );

    Batch(
        batch::decode_t,
        const generate::v2::Batch& pb_batch,
        CacheManager<DType, IdType>* cache_manager_,
        const HuggingfaceTokenizer& tokenizer
    );

    inline size_t size() const noexcept {
        return request_ids.size();
    }

    inline bool empty() const noexcept {
        return request_ids.empty();
    }

    inline bool is_prefill() const noexcept {
        bool res = all_tokens[0].size() == num_prompt_tokens[0];
        assert((res == !ragged_indptr.empty()));
        return res;
    }

    inline uint32_t get_max_tokens() const noexcept {
        return max_tokens;
    }

    /// \brief update page table and reallocate
    Batch<DType, IdType>&
    update(const std::vector<uint32_t>& last_output_tokens);

    Batch<DType, IdType>&
    filter(const ::google::protobuf::RepeatedField<uint64_t>& request_ids);

    Batch<DType, IdType>&
    extend(std::vector<std::reference_wrapper<Batch<DType, IdType>>>&);

    ::generate::v2::CachedBatch to_pb() const;

    std::tuple<
        ::google::protobuf::RepeatedPtrField<::generate::v2::Generation>,
        ::generate::v2::CachedBatch>
    to_pb(const HuggingfaceTokenizer&) const;
};

template<typename DType, typename IdType>
Batch<DType, IdType>::Batch(
    batch::prefill_t,
    size_t batch_max_tokens,
    uint64_t batch_id,
    size_t batch_size,
    std::vector<uint32_t>& input_tokens,
    uint64_t request_id,
    uint32_t max_new_tokens,
    CacheManager<DType, IdType>* cache_manager_,
    const HuggingfaceTokenizer& tokenizer
) :
    cache_manager(cache_manager_),
    page_table(cache_manager->empty_page_table(batch_size)),
    max_tokens(batch_max_tokens),
    id(batch_id),
    batch_size(batch_size),
    not_stopped(StoppingCriteria::length<DType, IdType>) {
    // for test model output
    std::vector<std::tuple<size_t, size_t>> needed_blocks_slots;
    size_t batch_need_blocks = 0;
    IdType indptr_iter = 0;

    request_ids.push_back(request_id);
    all_tokens.push_back(std::move(input_tokens));

    ragged_indptr.push_back(indptr_iter);
    indptr_iter += all_tokens.back().size();

    num_prompt_tokens.push_back(all_tokens.back().size());
    num_prompt_blocks.push_back(
        (all_tokens.back().size() + 1 + page_table.page_size - 1)
        / page_table.page_size
    );
    num_tokens += all_tokens.back().size();
    size_t req_max_tokens = all_tokens.back().size() + max_new_tokens;
    num_total_tokens.push_back(req_max_tokens);

    size_t need_slots = all_tokens.back().size() + 1;
    size_t need_blocks =
        (need_slots + page_table.page_size - 1) / page_table.page_size;
    size_t last_page_slot =
        need_slots - (need_blocks - 1) * page_table.page_size;

    page_table.indptr.push_back(batch_need_blocks);
    batch_need_blocks += need_blocks;

    page_table.last_page_len.push_back(last_page_slot);

    needed_blocks_slots.emplace_back(need_blocks, need_slots);
    // len(indptr_h) == batch_size + 1
    ragged_indptr.push_back(indptr_iter);
    page_table.indptr.push_back(batch_need_blocks);

    //todo check
    cache_manager->allocate(
        needed_blocks_slots,
        batch_need_blocks,
        page_table.indices,
        indices_2d,
        page_table.indptr,
        page_table.last_page_len,
        &page_table.indices_h,
        &page_table.indptr_h,
        &page_table.last_page_len_h,
        &pinned_memory_idx
    );
}

template<typename DType, typename IdType>
Batch<DType, IdType>::Batch(
    batch::prefill_t,
    const generate::v2::Batch& pb_batch,
    CacheManager<DType, IdType>* cache_manager_,
    const HuggingfaceTokenizer& tokenizer
) :
    cache_manager(cache_manager_),
    page_table(cache_manager->empty_page_table(pb_batch.size())),
    max_tokens(pb_batch.max_tokens()),
    id(pb_batch.id()),
    batch_size(pb_batch.size()),
    not_stopped(StoppingCriteria::length<DType, IdType>) {
    std::vector<std::tuple<size_t, size_t>> needed_blocks_slots;
    size_t batch_need_blocks = 0;
    IdType indptr_iter = 0;

    for (size_t i = 0; i < batch_size; ++i) {
        auto& request = pb_batch.requests(i);
        // auto input_tokens = tokenizer.encode(request.inputs());
        auto input_tokens = std::vector<uint32_t>(
            request.input_tokens().begin(),
            request.input_tokens().end()
        );
        request_ids.push_back(request.id());
        all_tokens.push_back(std::move(input_tokens));

        ragged_indptr.push_back(indptr_iter);
        indptr_iter += all_tokens.back().size();

        num_prompt_tokens.push_back(all_tokens.back().size());
        num_prompt_blocks.push_back(
            (all_tokens.back().size() + 1 + page_table.page_size - 1)
            / page_table.page_size
        );
        num_tokens += all_tokens.back().size();
        size_t req_max_tokens = all_tokens.back().size()
            + request.stopping_parameters().max_new_tokens();
        num_total_tokens.push_back(req_max_tokens);

        size_t need_slots = all_tokens.back().size() + 1;
        size_t need_blocks =
            (need_slots + page_table.page_size - 1) / page_table.page_size;
        size_t last_page_slot =
            need_slots - (need_blocks - 1) * page_table.page_size;

        page_table.indptr.push_back(batch_need_blocks);
        batch_need_blocks += need_blocks;

        page_table.last_page_len.push_back(last_page_slot);

        needed_blocks_slots.emplace_back(need_blocks, need_slots);
    }
    // len(indptr_h) == batch_size + 1
    ragged_indptr.push_back(indptr_iter);
    page_table.indptr.push_back(batch_need_blocks);

    //todo check
    cache_manager->allocate(
        needed_blocks_slots,
        batch_need_blocks,
        page_table.indices,
        indices_2d,
        page_table.indptr,
        page_table.last_page_len,
        &page_table.indices_h,
        &page_table.indptr_h,
        &page_table.last_page_len_h,
        &pinned_memory_idx
    );
}

template<typename DType, typename IdType>
Batch<DType, IdType>&
Batch<DType, IdType>::update(const std::vector<uint32_t>& last_output_tokens) {
    for (size_t i = 0; i < batch_size; ++i) {
        all_tokens[i].push_back(last_output_tokens[i]);
    }
    cache_manager->append(
        page_table.indices,
        this->indices_2d,
        page_table.indptr,
        page_table.last_page_len,
        &page_table.indices_h,
        &page_table.indptr_h,
        &page_table.last_page_len_h,
        pinned_memory_idx
    );
    return *this;
}

template<typename DType, typename IdType>
Batch<DType, IdType>& Batch<DType, IdType>::filter(
    const ::google::protobuf::RepeatedField<uint64_t>& pb_request_ids
) {
    std::vector<uint64_t> new_idx;
    std::vector<uint64_t> filtered_out_request_ids;
    for (size_t i = 0; i < request_ids.size(); ++i) {
        if (std::find(
                pb_request_ids.begin(),
                pb_request_ids.end(),
                request_ids[i]
            )
            != pb_request_ids.end()) {
            new_idx.push_back(i);
        } else {
            filtered_out_request_ids.push_back(request_ids[i]);
        }
    }
    BZ_INFO(
        "Batch[{}] Filtered out requests: [{}]",
        this->id,
        ::fmt::join(filtered_out_request_ids, ",")
    );

    size_t new_size = new_idx.size();
    batch_size = new_size;
    page_table.batch_size = new_size;
    // no need to filter
    if (new_size == this->size())
        return *this;

    if (new_size == 0) {
        throw std::runtime_error("Call filter batch with empty request_ids");
    }

    size_t cnt = 0;
    std::vector<uint64_t> new_request_ids(new_size);
    std::vector<std::vector<uint32_t>> new_all_tokens;

    IdType num_pages = 0;
    page_table.indptr = {0};
    page_table.indices.clear();
    for (auto idx : new_idx) {
        new_request_ids[cnt] = request_ids[idx];
        new_all_tokens.emplace_back(std::move(all_tokens[idx]));
        if (idx != cnt) {
            num_prompt_tokens[cnt] = num_prompt_tokens[idx];
            num_prompt_blocks[cnt] = num_prompt_blocks[idx];
            num_total_tokens[cnt] = num_total_tokens[idx];
            // page table
            std::swap(indices_2d[cnt], indices_2d[idx]);
            std::swap(
                page_table.last_page_len[cnt],
                page_table.last_page_len[idx]
            );
        }
        page_table.indices.insert(
            page_table.indices.end(),
            indices_2d[cnt].begin(),
            indices_2d[cnt].end()
        );
        page_table.indptr.push_back(num_pages + indices_2d[cnt].size());
        num_pages += indices_2d[cnt].size();
        cnt++;
    }

    for (; cnt < request_ids.size(); ++cnt) {
        cache_manager->free(indices_2d[cnt]);
        max_tokens -= indices_2d[cnt].size() * page_table.page_size;
        if (page_table.last_page_len[cnt] == 1) {
            // we minus page_size more tokens
            max_tokens += page_table.page_size;
        }
    }
    if ((int32_t)(max_tokens) < 0) {
        BZ_FATAL("current max_tokens={}", (int32_t)(max_tokens));
    }
    this->request_ids = std::move(new_request_ids);
    this->all_tokens = std::move(new_all_tokens);

    num_prompt_tokens.erase(
        num_prompt_tokens.begin() + new_size,
        num_prompt_tokens.end()
    );
    num_prompt_blocks.erase(
        num_prompt_blocks.begin() + new_size,
        num_prompt_blocks.end()
    );
    num_total_tokens.erase(
        num_total_tokens.begin() + new_size,
        num_total_tokens.end()
    );

    indices_2d.erase(indices_2d.begin() + new_size, indices_2d.end());
    page_table.last_page_len.erase(
        page_table.last_page_len.begin() + new_size,
        page_table.last_page_len.end()
    );

    assert((ragged_indptr.empty()));

    /// \bug forget to flush page table to pinned memory
    assert((page_table.indptr.back() == (int)page_table.indices.size()));
    std::copy(
        page_table.indices.begin(),
        page_table.indices.end(),
        page_table.indices_h
    );
    page_table.indptr_h = page_table.indices_h + page_table.indices.size();
    std::copy(
        page_table.indptr.begin(),
        page_table.indptr.end(),
        page_table.indptr_h
    );
    page_table.last_page_len_h = page_table.indptr_h + page_table.indptr.size();
    std::copy(
        page_table.last_page_len.begin(),
        page_table.last_page_len.end(),
        page_table.last_page_len_h
    );
    return *this;
}

template<typename DType, typename IdType>
std::tuple<
    ::google::protobuf::RepeatedPtrField<::generate::v2::Generation>,
    ::generate::v2::CachedBatch>
Batch<DType, IdType>::to_pb(const HuggingfaceTokenizer& tokenizer) const {
    ::google::protobuf::RepeatedPtrField<::generate::v2::Generation>
        pb_generations;
    // GeneratedText
    std::vector<std::string> generated_texts(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        if (!this->not_stopped(*this, i)) {
            generated_texts[i] = tokenizer.decode(all_tokens[i]);
        }
    }
    // CachedBatch
    ::generate::v2::CachedBatch pb_cached_batch;
    pb_cached_batch.set_id(this->id);
    pb_cached_batch.set_size(this->size());
    pb_cached_batch.set_max_tokens(this->max_tokens);
    for (size_t i = 0; i < this->size(); ++i) {
        pb_cached_batch.add_request_ids(request_ids[i]);
        ::generate::v2::Generation generation;
        generation.set_request_id(request_ids[i]);
        ::generate::v2::Tokens token;
        token.add_ids(all_tokens[i].back());
        token.add_texts(tokenizer.decode(all_tokens[i].back()));
        *generation.mutable_tokens() = token;
        ::generate::v2::GeneratedText generated_text;
        if (!generated_texts[i].empty()) {
            // BZ_DEBUG("Request[{}] generated_texts: {}", request_ids[i], generated_texts[i].c_str());
            *generated_text.mutable_text() = std::move(generated_texts[i]);
            generated_text.set_finish_reason(
                generate::v2::FINISH_REASON_LENGTH
            );
            generated_text.set_generated_tokens(
                all_tokens[i].size() - num_prompt_tokens[i]
            );
            *generation.mutable_generated_text() = generated_text;
        }
        *pb_generations.Add() = std::move(generation);
    }

    return {std::move(pb_generations), std::move(pb_cached_batch)};
}

template<typename DType, typename IdType>
::generate::v2::CachedBatch Batch<DType, IdType>::to_pb() const {
    // CachedBatch
    ::generate::v2::CachedBatch pb_cached_batch;
    pb_cached_batch.set_id(this->id);
    pb_cached_batch.set_size(this->size());
    pb_cached_batch.set_max_tokens(this->max_tokens);
    std::for_each(
        request_ids.cbegin(),
        request_ids.cend(),
        [&pb_cached_batch](const int64_t i) {
            pb_cached_batch.add_request_ids(i);
        }
    );
    return pb_cached_batch;
}

template<typename DType, typename IdType>
Batch<DType, IdType>& Batch<DType, IdType>::extend(
    std::vector<std::reference_wrapper<Batch<DType, IdType>>>& iterable
) {
    for (const Batch<DType, IdType>& iter : iterable) {
        max_tokens += iter.max_tokens;
        // request
        request_ids.insert(
            request_ids.end(),
            iter.request_ids.begin(),
            iter.request_ids.end()
        );
        all_tokens.insert(
            all_tokens.end(),
            iter.all_tokens.begin(),
            iter.all_tokens.end()
        );
        num_total_tokens.insert(
            num_total_tokens.end(),
            iter.num_total_tokens.begin(),
            iter.num_total_tokens.end()
        );
        num_prompt_tokens.insert(
            num_prompt_tokens.end(),
            iter.num_prompt_tokens.begin(),
            iter.num_prompt_tokens.end()
        );
        num_prompt_blocks.insert(
            num_prompt_blocks.end(),
            iter.num_prompt_blocks.begin(),
            iter.num_prompt_blocks.end()
        );
        // batch size
        batch_size += iter.batch_size;
        // page table
        page_table.batch_size += iter.batch_size;
        page_table.indices.insert(
            page_table.indices.end(),
            iter.page_table.indices.begin(),
            iter.page_table.indices.end()
        );
        indices_2d.insert(
            indices_2d.end(),
            iter.indices_2d.begin(),
            iter.indices_2d.end()
        );
        IdType indptr_tail = page_table.indptr.back();
        page_table.indptr.pop_back();
        for (auto ind : iter.page_table.indptr) {
            page_table.indptr.emplace_back(ind + indptr_tail);
        }
        page_table.last_page_len.insert(
            page_table.last_page_len.end(),
            iter.page_table.last_page_len.begin(),
            iter.page_table.last_page_len.end()
        );
        cache_manager->free_block_meta(iter.pinned_memory_idx);
    }
    assert((this->indices_2d.size() == batch_size));
    assert((page_table.indptr.size() == batch_size + 1));
    assert((page_table.last_page_len.size() == batch_size));

    // Flush to pinned memory
    IdType* cummulated_pinned_memory_idx = this->page_table.indices_h;

    std::copy(
        page_table.indices.begin(),
        page_table.indices.end(),
        cummulated_pinned_memory_idx
    );
    cummulated_pinned_memory_idx += page_table.indices.size();
    this->page_table.indptr_h = cummulated_pinned_memory_idx;

    std::copy(
        page_table.indptr.begin(),
        page_table.indptr.end(),
        cummulated_pinned_memory_idx
    );
    cummulated_pinned_memory_idx += page_table.indptr.size();
    this->page_table.last_page_len_h = cummulated_pinned_memory_idx;

    std::copy(
        page_table.last_page_len.begin(),
        page_table.last_page_len.end(),
        cummulated_pinned_memory_idx
    );

    BZ_DEBUG(
        "Extend Batch[{}] Requests[{}]",
        this->id,
        ::fmt::join(this->request_ids, ",")
    );

    return *this;
}

}  // namespace blitz
