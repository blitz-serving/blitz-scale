#include "model_loader.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <future>
#include <thread>
#include <vector>

#include "include/logger.hpp"
#include "include/types.hpp"
#include "util/cuda_utils.h"

namespace blitz::model {

ModelLoader::ModelLoader(
    const rank_t& device_,
    void* weight_segment_,
    size_t weight_segment_size_in_bytes_
) :
    device(device_),
    host_segment_size_in_bytes(weight_segment_size_in_bytes_),
    weight_segment((char*)weight_segment_),
    weight_segment_size_in_bytes(weight_segment_size_in_bytes_) {
    cudaHostAlloc(
        &host_weight_segment,
        host_segment_size_in_bytes,
        cudaHostAllocDefault
    );
}

ModelLoader::~ModelLoader() {
    cudaFree(weight_segment);
    cudaFreeHost(host_weight_segment);
}

void ModelLoader::load_model_host_fast(const std::string& file_name) {
    mutex.lock();
    int fd = open(file_name.c_str(), O_DIRECT | O_RDONLY);
    if (fd == -1) {
        BZ_FATAL("Failed to open file: {}", file_name);
    }
    struct stat st;
    fstat(fd, &st);
    const size_t file_size = st.st_size;
    assert((file_size == host_segment_size_in_bytes));

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::future<int>> as;
    for (size_t i = 0; i < nthrd; ++i) {
        size_t total_chunks =
            (file_size + chunk_size_in_bytes - 1) / chunk_size_in_bytes;
        size_t chunk_per_thrd = (i < nthrd - 1)
            ? (total_chunks + nthrd - 1) / nthrd
            : total_chunks / nthrd;
        size_t partition_offset = i * chunk_per_thrd * chunk_size_in_bytes;

        as.emplace_back(std::async(
            std::launch::async,
            [this, file_size, fd, chunk_per_thrd, partition_offset]() {
                size_t offset = partition_offset;
                for (size_t i = 0; i < chunk_per_thrd; ++i) {
                    size_t nbytes =
                        std::min(chunk_size_in_bytes, file_size - offset);
                    ssize_t bytes_read = pread(
                        fd,
                        this->host_weight_segment + offset,
                        nbytes,
                        offset
                    );

                    if (bytes_read < 0) {
                        BZ_ERROR(
                            "Read chunk [{},{}] w/ errno: {} {}",
                            offset,
                            offset + nbytes,
                            errno,
                            strerror(errno)
                        );
                        return -1;
                    } else if ((size_t)bytes_read != nbytes) {
                        BZ_ERROR(
                            "Read chunk [{},{}] for {} bytes, but read {} bytes",
                            offset,
                            offset + nbytes,
                            bytes_read
                        );
                        return -2;
                    }

                    offset += bytes_read;
                }
                return 0;
            }
        ));
    }

    for (auto& a : as) {
        a.wait();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(stop - start);
    BZ_INFO(
        "Load model weight to host takes {} ms, effective bw {} GBps",
        duration,
        (double)file_size / duration.count() / 1000 / 1000
    );

    valid.store(true, std::memory_order::release);
    close(fd);
    mutex.unlock();
}

void ModelLoader::load_model_host_fast(
    const std::string& file_name,
    loader::overlap_ssd_gpu_t
) {
    mutex.lock();
    int fd = open(file_name.c_str(), O_DIRECT | O_RDONLY);
    if (fd == -1) {
        BZ_ERROR("Failed to open file: {}", file_name);
    }
    struct stat st;
    fstat(fd, &st);
    const size_t file_size = st.st_size;
    assert((file_size == host_segment_size_in_bytes));

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::atomic_size_t> offsets(nthrd);
    std::vector<std::future<int>> as;
    std::vector<std::future<void>> bs;
    for (size_t i = 0; i < nthrd; ++i) {
        size_t total_chunks =
            (file_size + chunk_size_in_bytes - 1) / chunk_size_in_bytes;
        size_t chunk_per_thrd = (i < nthrd - 1)
            ? (total_chunks + nthrd - 1) / nthrd
            : total_chunks / nthrd;
        /// @todo @bug The offset of the last thread will be WRONG if nthrd can't divide the total chunks.
        size_t partition_offset = i * chunk_per_thrd * chunk_size_in_bytes;
        offsets[i].store(partition_offset, std::memory_order::release);

        as.emplace_back(std::async(
            std::launch::async,
            [this,
             file_size,
             fd,
             chunk_per_thrd,
             partition_offset,
             &offsets,
             i]() {
                size_t offset = partition_offset;
                auto& shared_offset = offsets[i];
                for (size_t i = 0; i < chunk_per_thrd; ++i) {
                    size_t nbytes =
                        std::min(chunk_size_in_bytes, file_size - offset);
                    ssize_t bytes_read = pread(
                        fd,
                        this->host_weight_segment + offset,
                        nbytes,
                        offset
                    );

                    if (bytes_read < 0) {
                        BZ_ERROR(
                            "Read chunk [{},{}] w/ errno: {} {}",
                            offset,
                            offset + nbytes,
                            errno,
                            strerror(errno)
                        );
                        return -1;
                    } else if ((size_t)bytes_read != nbytes) {
                        BZ_ERROR(
                            "Read chunk [{},{}] for {} bytes, but read {} bytes",
                            offset,
                            offset + nbytes,
                            bytes_read
                        );
                        return -2;
                    }

                    offset += bytes_read;
                    shared_offset.store(offset, std::memory_order::release);
                }
                return 0;
            }
        ));
        /// \note end_offset is one of {next partition_offset, eof_offset}
        size_t end_offset = std::min(
            file_size,
            partition_offset + chunk_per_thrd * chunk_size_in_bytes
        );
        bs.emplace_back(std::async(
            std::launch::async,
            [this, partition_offset, end_offset, &offsets, i]() {
                CUDA_CHECK(cudaSetDevice(device));
                size_t offset = partition_offset;
                auto& shared_offset = offsets[i];
                size_t tmp;
                while (offset < end_offset) {
                    while ((tmp =
                                shared_offset.load(std::memory_order::relaxed))
                           <= offset) {
                        std::this_thread::yield();
                    }
                    CUDA_CHECK(cudaMemcpy(
                        this->weight_segment + offset,
                        this->host_weight_segment + offset,
                        tmp - offset,
                        cudaMemcpyHostToDevice
                    ));
                    offset = tmp;
                }
            }
        ));
    }

    for (auto& a : as) {
        a.wait();
    }
    for (auto& b : bs) {
        b.wait();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(stop - start);
    BZ_INFO(
        "Load model weight from ssd to gpu takes {} ms, effective bw {} GBps",
        duration,
        (double)file_size / duration.count() / 1000 / 1000
    );

    valid.store(true, std::memory_order::release);
    close(fd);
    mutex.unlock();
}

void ModelLoader::load_model_gpu() {
    mutex.lock();
    /// \todo Add Publisher & Subscriber
    // assert(valid);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::future<void>> bs;
    for (size_t i = 0; i < nthrd; ++i) {
        size_t total_chunks =
            (weight_segment_size_in_bytes + chunk_size_in_bytes - 1)
            / chunk_size_in_bytes;
        size_t chunk_per_thrd = (i < nthrd - 1)
            ? (total_chunks + nthrd - 1) / nthrd
            : total_chunks / nthrd;
        size_t partition_offset = i * chunk_per_thrd * chunk_size_in_bytes;

        bs.emplace_back(std::async(
            std::launch::async,
            [this, partition_offset, chunk_per_thrd]() {
                CUDA_CHECK(cudaSetDevice(device));
                size_t offset = partition_offset;
                for (size_t i = 0; i < chunk_per_thrd; ++i) {
                    cudaMemcpyAsync(
                        this->weight_segment + offset,
                        this->host_weight_segment + offset,
                        (offset + chunk_size_in_bytes
                         <= weight_segment_size_in_bytes)
                            ? chunk_size_in_bytes
                            : weight_segment_size_in_bytes - offset,
                        cudaMemcpyHostToDevice
                    );
                }
                CUDA_CHECK(cudaStreamSynchronize(0));
            }
        ));
    }

    for (auto& b : bs) {
        b.wait();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(stop - start);
    BZ_INFO(
        "Load model weight from ssd to gpu takes {} ms, effective bw {} GBps",
        duration,
        (double)weight_segment_size_in_bytes / duration.count() / 1000 / 1000
    );

    mutex.unlock();
}

}  // namespace blitz::model