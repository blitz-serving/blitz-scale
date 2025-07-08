#include "rms_norm.h"

#include <flashinfer/norm.cuh>

namespace blitz::kernel {

template<typename DType>
void rms_norm(
    DType* input,
    DType* weight,
    DType* output,
    uint32_t batch_size,
    uint32_t hidden_size,
    float eps
) {
    cudaError_t status = flashinfer::norm::RMSNorm<DType>(
                            input, weight, output,
                            batch_size, hidden_size, eps);
    if (status != cudaSuccess) {
        throw std::runtime_error("rmsNorm error: " + std::string(cudaGetErrorString(status)));
    }
}

template<typename DType>
void add_rms_norm(
    DType* input,
    DType* residual,
    DType* weight,
    uint32_t batch_size,
    uint32_t hidden_size,
    float eps
) {
    cudaError_t status = flashinfer::norm::FusedAddRMSNorm<DType>(
                            input, residual, weight,
                            batch_size, hidden_size, eps);
    if (status != cudaSuccess) {
        throw std::runtime_error("fusedAddRmsNorm error: " + std::string(cudaGetErrorString(status)));
    }
}

// instantiation
template void rms_norm<half>(
    half*, half*, half*, uint32_t, uint32_t, float
);

template void add_rms_norm<half>(
    half*, half*, half*, uint32_t, uint32_t, float
);

}