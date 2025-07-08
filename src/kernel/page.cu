#include "page.h"

#include <flashinfer/page.cuh>
#include <flashinfer/vec_dtypes.cuh>
#include <assert.h>

namespace blitz::kernel {

/*!
 * \brief CUDA kernel to store new keys/values to the paged key-value cache in the prefill phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param ragged_indptr The indptr array of the appended ragged tensor
 */
template <typename DType, typename IdType>
__global__ void StorePagedKVCachePrefillKernel(
    flashinfer::paged_kv_t<DType, IdType> paged_kv,
    DType* __restrict__ key, DType* __restrict__ value,
    IdType* __restrict__ ragged_indptr
) {
    constexpr uint32_t vec_size = 8;
    // a warp for 1 token (aka. num_heads * head_size)
    uint32_t tx = threadIdx.x, head_idx = threadIdx.y;
    uint32_t head_dim = paged_kv.head_dim;
    uint32_t num_heads = paged_kv.num_heads;
    uint32_t page_size = paged_kv.page_size;
    // seq idx within a batch, ranging [0 .. batch_size - 1]
    uint32_t batch_idx = blockIdx.y;
    // token within a page, ranging [0 .. page_size - 1]
    uint32_t entry_idx = blockIdx.x;
    
    uint32_t seq_len =
        (paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
        paged_kv.last_page_len[batch_idx];
    uint32_t append_seq_len = ragged_indptr[batch_idx + 1] - ragged_indptr[batch_idx];
    // CUDA assert
    assert((seq_len >= append_seq_len));

#pragma unroll
    for (uint32_t page_seq_idx = entry_idx;
            page_seq_idx < append_seq_len;
            page_seq_idx += page_size
    ) {
        // idx of page_id
        uint32_t page_iter = paged_kv.indptr[batch_idx] + page_seq_idx / page_size;
        // get page_id, then calculate page_offset
        DType* k_ptr = paged_kv.get_k_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
        DType* v_ptr = paged_kv.get_v_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
        flashinfer::vec_t<DType, vec_size>::memcpy(
            k_ptr,
            key + ((ragged_indptr[batch_idx] + page_seq_idx) * num_heads + head_idx) * head_dim + tx * vec_size);
        flashinfer::vec_t<DType, vec_size>::memcpy(
            v_ptr,
            value + ((ragged_indptr[batch_idx] + page_seq_idx) * num_heads + head_idx) * head_dim + tx * vec_size);
    }
}

template<typename DType, typename IdType>
void store_prefill_kv_cache(
    PageTable<DType, IdType>& page_table,
    DType* k_data,
    DType* v_data,
    IdType* ragged_indptr
) {
    flashinfer::paged_kv_t<DType, IdType> paged_kv(
        page_table.num_heads, page_table.page_size, page_table.head_size,
        page_table.batch_size, flashinfer::QKVLayout::kNHD,
        page_table.k_data, page_table.v_data,// empty data
        page_table.indices_d, page_table.indptr_d, page_table.last_page_len_d
    );
    // cudaError_t status = flashinfer::AppendPagedKVCache(
    //     paged_kv,
    //     k_data, v_data, // real data
    //     ragged_indptr
    // );
    cudaError_t status;
    {
        constexpr uint32_t vec_size = 8;
        dim3 nblks(page_table.page_size, page_table.batch_size);
        dim3 nthrs(page_table.head_size / vec_size, page_table.num_heads);
        auto kernel = StorePagedKVCachePrefillKernel<DType, IdType>;    
        void* args[] = {(void*)&paged_kv, (void*)&k_data, (void*)&v_data, (void*)&ragged_indptr};
        status = cudaLaunchKernel((void*)kernel, nblks, nthrs, args);    
    }
    if (status != cudaSuccess) {
        throw std::runtime_error("AppendPagedKVCache error: " + std::string(cudaGetErrorString(status)));
    }
}

template<typename DType, typename IdType>
void append_decode_kv_cache(
    PageTable<DType, IdType>& page_table,
    DType* k_data,
    DType* v_data
) {
    flashinfer::paged_kv_t<DType, IdType> paged_kv(
        page_table.num_heads, page_table.page_size, page_table.head_size,
        page_table.batch_size, flashinfer::QKVLayout::kNHD,
        page_table.k_data, page_table.v_data,
        page_table.indices_d, page_table.indptr_d, page_table.last_page_len_d
    );
    cudaError_t status = flashinfer::AppendPagedKVCacheDecode(
        paged_kv,
        k_data, v_data // real data
    );
    if (status != cudaSuccess) {
        throw std::runtime_error("AppendPagedKVCacheDecode error: " + std::string(cudaGetErrorString(status)));
    }
}

// instantiation
template void store_prefill_kv_cache<half, int32_t>(
    PageTable<half, int32_t>&, half*, half*, int32_t*
);

template void append_decode_kv_cache<half, int32_t>(
    PageTable<half, int32_t>&, half*, half*
);

}