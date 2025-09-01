#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/device_vector.h>

namespace liquidity {
namespace cuda {

// CUDA kernel for Amihud illiquidity
__global__ void amihud_kernel(
    const double* __restrict__ prices,
    const double* __restrict__ volumes,
    double* __restrict__ output,
    int n,
    int window_size
);

// CUDA kernel for duration calculation
__global__ void duration_kernel(
    const int64_t* __restrict__ timestamps,
    double* __restrict__ durations,
    int n
);

// CUDA kernel for hazard rate (Weibull)
__global__ void hazard_kernel(
    const double* __restrict__ durations,
    double* __restrict__ hazard_rates,
    double lambda,
    double shape,
    int n
);

// CUDA kernel for identifying holes
__global__ void holes_kernel(
    const double* __restrict__ durations,
    int32_t* __restrict__ holes,
    double threshold,
    int n
);

// Multi-GPU coordinator class
class MultiGPUProcessor {
public:
    MultiGPUProcessor(int num_gpus = -1);
    ~MultiGPUProcessor();

    // Process trades across multiple GPUs
    void process_trades_multi_gpu(
        const std::vector<TradeData>& trades,
        std::vector<double>& amihud_out,
        std::vector<double>& durations_out,
        std::vector<double>& hazard_out
    );

    // Process quotes across multiple GPUs
    void process_quotes_multi_gpu(
        const std::vector<QuoteData>& quotes,
        std::vector<double>& spreads_out,
        std::vector<double>& depths_out
    );

private:
    struct GPUContext {
        int device_id;
        cudaStream_t stream;
        cublasHandle_t cublas_handle;
        
        // Device memory pools
        thrust::device_vector<double> d_prices;
        thrust::device_vector<double> d_volumes;
        thrust::device_vector<int64_t> d_timestamps;
        thrust::device_vector<double> d_output;
    };

    std::vector<std::unique_ptr<GPUContext>> gpu_contexts_;
    int num_gpus_;
    
    void initialize_gpu_contexts();
    void distribute_work(size_t total_size, std::vector<size_t>& work_sizes);
};

// Optimized memory pool for high-frequency data
class CUDAMemoryPool {
public:
    CUDAMemoryPool(size_t initial_size);
    ~CUDAMemoryPool();

    void* allocate(size_t size);
    void deallocate(void* ptr);
    void reset();

private:
    cudaMemPool_t mem_pool_;
    size_t current_size_;
    size_t max_size_;
};

} // namespace cuda
} // namespace liquidity
