#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

#include "include/types.hpp"

namespace blitz::model {

// clang-format off
namespace loader {
struct overlap_ssd_gpu_t { explicit overlap_ssd_gpu_t() = default; };


inline constexpr overlap_ssd_gpu_t overlap_ssd_gpu {};

}  // namespace loader

// clang-format on

class ModelLoader {
  public:
    ModelLoader(
        const rank_t& device_,
        void* weight_segment_,
        size_t weight_segment_size_in_bytes_
    );

    ~ModelLoader();

    /// \brief utilize full IOPS to load model from SSD to host
    /// \param file_name a compact ".dangertensor" file
    void load_model_host_fast(const std::string& file_name);

    /// \brief utilize full IOPS to load model from SSD to host
    ///        overlap loading from ssd to host && host to gpu
    /// \param file_name a compact ".dangertensor" file
    void load_model_host_fast(
        const std::string& file_name,
        loader::overlap_ssd_gpu_t
    );

    /// \brief load model from host to GPU
    void load_model_gpu();

    /// \brief host weight buffer holds all parameters
    bool is_ready() const noexcept {
        return valid.load(std::memory_order::acquire);
    }

  private:
    // CUDA device
    const rank_t& device;
    // pinned memory
    char* host_weight_segment;
    size_t host_segment_size_in_bytes;
    // gpu memory
    char* weight_segment;
    size_t weight_segment_size_in_bytes;

    // lock
    std::mutex mutex;
    // valid flag
    std::atomic_bool valid = false;
    // concurrency
    static constexpr size_t nthrd = 2;
    static constexpr size_t chunk_size_in_bytes = 64 * 1024 * 1024;
};

}  // namespace blitz::model