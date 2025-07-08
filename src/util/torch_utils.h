#pragma once

#include <cuda_fp16.h>
#include <torch/torch.h>

#include <fstream>
#include <sstream>
#include <vector>

namespace blitz::util {

template<typename T>
inline torch::ScalarType getTorchScalarType() {
    if (std::is_same<T, float>::value) {
        return torch::kFloat;
    } else if (std::is_same<T, half>::value) {
        return torch::kHalf;
    } else if (std::is_same<T, int64_t>::value) {
        return torch::kInt64;
    } else {
        throw std::runtime_error("Unsupported type");
    }
}

inline void* convertTensorToRawPtr(torch::Tensor& tensor) {
    if (tensor.scalar_type() == torch::kFloat) {
        return tensor.data_ptr<float>();
    } else if (tensor.scalar_type() == torch::kHalf) {
        return tensor.data_ptr<at::Half>();
    } else if (tensor.scalar_type() == torch::kInt64) {
        return tensor.data_ptr<int64_t>();
    } else {
        throw std::runtime_error("Unsupported type");
    }
}

inline size_t getTensorSizeInBytes(torch::Tensor tensor) {
    return tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
}

template<typename DType>
auto initDummyTensor = [](DType* addr, int64_t numel) {
    // torch::from_blob: Exposes the given data as a Tensor without taking ownership of the original data.
    // So when tmp is destructed, the memory won't be freed
    ::torch::Tensor tmp = ::torch::from_blob(
        addr,
        {numel},
        ::torch::TensorOptions().dtype(getTorchScalarType<DType>()).device(torch::kCUDA)
    );
    tmp.uniform_(-1e-3, 1e-3);
};

template<typename DType>
void load_tensor_from_file(
    DType** iter,
    const std::string& filename,
    size_t size_in_bytes,
    DType* host_weight_segment
) {
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file");
    }

    // Move to the starting position
    file.seekg(0);

    // Read the specified number of bytes
    if (!file.read(reinterpret_cast<char*>(host_weight_segment), size_in_bytes)) {
        throw std::runtime_error("Error reading from file.");
    }
    auto err = cudaMemcpy(*iter, reinterpret_cast<char*>(host_weight_segment), size_in_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data to device");
    }
    file.close();
}
}  // namespace blitz::util
