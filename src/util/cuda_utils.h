#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../include/logger.hpp"
#include "fmt/core.h"
#include "fmt/format.h"

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t result = cmd; \
        if (result != cudaSuccess) { \
            BZ_ERROR("CUDA error '{}': ({}) {}\n", #cmd, (int)result, cudaGetErrorString(result)); \
            exit(-1); \
        } \
    } while (0)

template<typename T>
inline void dumpPrint(const T* d_buf, size_t size, const char* name, cudaStream_t stream = 0) {
    T* h_buf = new T[size];
    BZ_INFO("Dump buffer {} @ {}: ", name, ::fmt::ptr(d_buf));
    cudaStreamSynchronize(stream);
    CUDA_CHECK(cudaMemcpy(h_buf, d_buf, sizeof(T) * size, cudaMemcpyDeviceToHost));
    printf("[0x%08lx ... 0x%08lx]\n", *(uintptr_t*)h_buf, *((uintptr_t*)(h_buf + size) - 1));
    delete[] h_buf;
}

inline void syncAndCheck(const char* const file, int const line, bool force_check = false) {
#ifdef DEBUG
    force_check = true;
#endif
    if (force_check) {
        cudaDeviceSynchronize();
        cudaError_t result = cudaGetLastError();
        if (result) {
            auto err_msg = ::fmt::format("[ST] CUDA runtime error: {} {}:{}\n", cudaGetErrorString(result), file, line);
            BZ_ERROR("CUDA runtime error: {}", cudaGetErrorString(result));
            throw std::runtime_error(err_msg);
        }
    }
}

#define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__, false)
#define sync_check_cuda_error_force() syncAndCheck(__FILE__, __LINE__, true)

// Some stuff for indexing into an 1-D array
#define INDEX_2D(dim1, dim2, index1, index2) (((int64_t)index1) * (dim2) + (index2))
#define INDEX_3D(dim1, dim2, dim3, index1, index2, index3) \
    (((int64_t)index1) * (dim2) * (dim3) + ((int64_t)index2) * (dim3) + (index3))
#define INDEX_4D(dim1, dim2, dim3, dim4, index1, index2, index3, index4) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) + ((int64_t)index2) * (dim3) * (dim4) + ((int64_t)index3) * (dim4) \
     + (index4))
#define INDEX_5D(dim1, dim2, dim3, dim4, dim5, index1, index2, index3, index4, index5) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) * (dim5) + ((int64_t)index2) * (dim3) * (dim4) * (dim5) \
     + ((int64_t)index3) * (dim4) * (dim5) + (index4) * (dim5) + (index5))

// A tiny stuff that supports remalloc on GPU
template<typename T>
struct RemallocableArray {
    T* ptr {nullptr};
    size_t size = 0;

    /// @bug blame to zdy, RemallocableArray is trivially default constructed,
    /// if no default initializers are provided.
    /// which means the default constructor does nothing,
    /// other than initializing ptr to nullptr
    /// https://en.cppreference.com/w/cpp/language/default_constructor
    RemallocableArray() = default;

    ~RemallocableArray() {
        if (ptr != nullptr) {
            CUDA_CHECK(cudaFree(ptr));
        }
    }

    /** @bug wrong: violates RAII
        RemallocableArray& operator=(const RemallocableArray<T>& other) {
            if (ptr != nullptr) {
                CUDA_CHECK(cudaFree(ptr));
            }
            ptr = other.ptr;
            size = other.size;
            return *this;
        }
    */
    RemallocableArray(const RemallocableArray<T>& other) = delete;

    RemallocableArray(RemallocableArray<T>&& other) : ptr(other.ptr), size(other.size) {
        other.ptr = nullptr;
        other.size = 0;
    }

    RemallocableArray& operator=(const RemallocableArray<T>&) = delete;

    RemallocableArray& operator=(RemallocableArray<T>&& other) {
        if (this == &other) {
            throw std::runtime_error("RemallocableArra: move assign to self!");
        }
        // clear();
        // in case this == &other; rvalue ref is also REFERENCE !!!
        if (ptr != nullptr) {
            CUDA_CHECK(cudaFreeAsync(ptr, 0));
            ptr = nullptr;
            size = 0;
        }
        ptr = other.ptr;
        size = other.size;
        other.ptr = nullptr;
        other.size = 0;
        return *this;
    }

    void remalloc(size_t target_size) {
        if (target_size > size) {
            size_t new_size = size ? size * 2 : 64;
            while (new_size < target_size) {
                new_size *= 2;
            }
            if (ptr != nullptr) {
                CUDA_CHECK(cudaFreeAsync(ptr, 0));
            }
            CUDA_CHECK(cudaMallocAsync(&ptr, new_size * sizeof(T), 0));
            size = new_size;
        }
    }

    RemallocableArray& borrow(const RemallocableArray<T>& other) {
        if (this == &other) {
            throw std::runtime_error("RemallocableArra: borrow to self!");
        }
        /// @note borrow without ownership; aka Python-like `=`
        /// thus don't do cleanup; no RAII
        /// skip clear();
        ptr = other.ptr;
        size = other.size;
        return *this;
    }
};

template<typename T>
inline void printGpuArrayHelper(const T* array, int64_t size, const char* arr_name) {
    T* array_cpu = new T[size];
    CUDA_CHECK(cudaMemcpy(array_cpu, array, sizeof(T) * size, cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < size; i++) {
        printf("%f ", (float)array_cpu[i]);
    }
    printf("\n");
    delete[] array_cpu;
}

#define printGpuArray(array, size) printGpuArrayHelper(array, size, #array)

// A util to check cuda memory usage
inline int64_t cuda_memory_size() {
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    return total_byte - free_byte;
}

// CUDAFreeAtReturn - A tiny macro to call cudaFree when the point goes out of scope
template<typename PTR_T>
class CUDAFreeAtReturnHelper {
  private:
    PTR_T ptr;
    std::string pointer_name;

  public:
    CUDAFreeAtReturnHelper(PTR_T ptr, std::string pointer_name) : pointer_name(pointer_name) {
        this->ptr = ptr;
    }

    ~CUDAFreeAtReturnHelper() {
        if (ptr != nullptr) {
            cudaFree(ptr);
            cudaDeviceSynchronize();
            cudaError_t result = cudaGetLastError();
            if (result) {
                fprintf(stderr, "Error occured when freeing pointer %s\n", pointer_name.c_str());
                fprintf(
                    stderr,
                    "%s\n",
                    (std::string("[ST] CUDA runtime error: ") + cudaGetErrorString(result) + " " + __FILE__ + ":"
                     + std::to_string(__LINE__) + " \n")
                        .c_str()
                );
                exit(1);
            }
        }
    }
};

#define CUDA_FREE_AT_RETURN(ptr) CUDAFreeAtReturnHelper<decltype(ptr)> ptr##_cuda_free_at_return(ptr, #ptr)

template<typename T>
cudaDataType_t getCudaDataType() {
    if (std::is_same<T, half>::value) {
        return CUDA_R_16F;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        return CUDA_R_16BF;
    }
#endif
    else if (std::is_same<T, float>::value) {
        return CUDA_R_32F;
    } else {
        throw std::runtime_error("Cuda data type: Unsupported type");
    }
}

// TODO@ly: a cuda memory buddy system
template<typename T>
struct CudaMemPool {
    size_t capacity;
    size_t slot_size;  // 64 in forward pass

    CudaMemPool(size_t, size_t);
    bool Alloc(size_t);
    bool Free(size_t);
};