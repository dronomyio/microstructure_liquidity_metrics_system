// ==================== src/cpp/cuda_processor.cu ====================
#include "cuda_processor.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <iostream>

namespace liquidity {
namespace cuda {

// Kernel implementations
__global__ void amihud_kernel(
    const double* __restrict__ prices,
    const double* __restrict__ volumes,
    double* __restrict__ output,
    int n,
    int window_size
) {
    int window_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_windows = (n + window_size - 1) / window_size;
    
    if (window_id >= num_windows) return;
    
    int start = window_id * window_size;
    int end = min(start + window_size, n);
    
    if (end <= start + 1) {
        output[window_id] = 0.0;
        return;
    }
    
    // Calculate return
    double price_return = (prices[end-1] - prices[start]) / prices[start];
    
    // Calculate dollar volume
    double dollar_volume = 0.0;
    for (int i = start; i < end; ++i) {
        dollar_volume += prices[i] * volumes[i];
    }
    
    output[window_id] = fabs(price_return) / (dollar_volume / 1e6);
}

__global__ void duration_kernel(
    const int64_t* __restrict__ timestamps,
    double* __restrict__ durations,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n) return;
    
    durations[tid] = static_cast<double>(timestamps[tid+1] - timestamps[tid]) / 1e9;
}

__global__ void hazard_kernel(
    const double* __restrict__ durations,
    double* __restrict__ hazard_rates,
    double lambda,
    double shape,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n) return;
    
    double t = durations[tid];
    hazard_rates[tid] = (shape / lambda) * pow(t / lambda, shape - 1.0);
}

__global__ void holes_kernel(
    const double* __restrict__ durations,
    int32_t* __restrict__ holes,
    double threshold,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n) return;
    
    holes[tid] = (durations[tid] > threshold) ? 1 : 0;
}

// MultiGPUProcessor implementation
MultiGPUProcessor::MultiGPUProcessor(int num_gpus) {
    if (num_gpus == -1) {
        cudaGetDeviceCount(&num_gpus_);
    } else {
        num_gpus_ = num_gpus;
    }
    
    initialize_gpu_contexts();
}

MultiGPUProcessor::~MultiGPUProcessor() {
    for (auto& ctx : gpu_contexts_) {
        cudaSetDevice(ctx->device_id);
        cudaStreamDestroy(ctx->stream);
        cublasDestroy(ctx->cublas_handle);
    }
}

void MultiGPUProcessor::initialize_gpu_contexts() {
    gpu_contexts_.resize(num_gpus_);
    
    for (int i = 0; i < num_gpus_; ++i) {
        gpu_contexts_[i] = std::make_unique<GPUContext>();
        gpu_contexts_[i]->device_id = i;
        
        cudaSetDevice(i);
        cudaStreamCreate(&gpu_contexts_[i]->stream);
        cublasCreate(&gpu_contexts_[i]->cublas_handle);
        cublasSetStream(gpu_contexts_[i]->cublas_handle, gpu_contexts_[i]->stream);
    }
}

void MultiGPUProcessor::distribute_work(size_t total_size, std::vector<size_t>& work_sizes) {
    work_sizes.resize(num_gpus_);
    size_t base_size = total_size / num_gpus_;
    size_t remainder = total_size % num_gpus_;
    
    for (int i = 0; i < num_gpus_; ++i) {
        work_sizes[i] = base_size + (i < remainder ? 1 : 0);
    }
}

void MultiGPUProcessor::process_trades_multi_gpu(
    const std::vector<TradeData>& trades,
    std::vector<double>& amihud_out,
    std::vector<double>& durations_out,
    std::vector<double>& hazard_out
) {
    size_t n = trades.size();
    if (n == 0) return;
    
    std::vector<size_t> work_sizes;
    distribute_work(n, work_sizes);
    
    // Process on each GPU
    size_t offset = 0;
    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
        cudaSetDevice(gpu);
        auto& ctx = gpu_contexts_[gpu];
        size_t gpu_n = work_sizes[gpu];
        
        if (gpu_n == 0) continue;
        
        // Allocate and copy data
        ctx->d_prices.resize(gpu_n);
        ctx->d_volumes.resize(gpu_n);
        ctx->d_timestamps.resize(gpu_n);
        
        std::vector<double> h_prices(gpu_n), h_volumes(gpu_n);
        std::vector<int64_t> h_timestamps(gpu_n);
        
        for (size_t i = 0; i < gpu_n; ++i) {
            h_prices[i] = trades[offset + i].price;
            h_volumes[i] = trades[offset + i].volume;
            h_timestamps[i] = trades[offset + i].timestamp_ns;
        }
        
        thrust::copy(h_prices.begin(), h_prices.end(), ctx->d_prices.begin());
        thrust::copy(h_volumes.begin(), h_volumes.end(), ctx->d_volumes.begin());
        thrust::copy(h_timestamps.begin(), h_timestamps.end(), ctx->d_timestamps.begin());
        
        // Calculate Amihud
        int window_size = 390;
        int num_windows = (gpu_n + window_size - 1) / window_size;
        ctx->d_output.resize(num_windows);
        
        int block_size = 256;
        int grid_size = (num_windows + block_size - 1) / block_size;
        
        amihud_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
            thrust::raw_pointer_cast(ctx->d_prices.data()),
            thrust::raw_pointer_cast(ctx->d_volumes.data()),
            thrust::raw_pointer_cast(ctx->d_output.data()),
            gpu_n,
            window_size
        );
        
        // Calculate durations
        if (gpu_n > 1) {
            thrust::device_vector<double> d_durations(gpu_n - 1);
            grid_size = (gpu_n - 1 + block_size - 1) / block_size;
            
            duration_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
                thrust::raw_pointer_cast(ctx->d_timestamps.data()),
                thrust::raw_pointer_cast(d_durations.data()),
                gpu_n - 1
            );
            
            // Calculate hazard rates
            thrust::device_vector<double> d_hazard(gpu_n - 1);
            hazard_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
                thrust::raw_pointer_cast(d_durations.data()),
                thrust::raw_pointer_cast(d_hazard.data()),
                1.0, 1.0,  // lambda, shape parameters
                gpu_n - 1
            );
            
            // Copy results back
            cudaStreamSynchronize(ctx->stream);
            
            std::vector<double> h_durations(gpu_n - 1), h_hazard(gpu_n - 1);
            thrust::copy(d_durations.begin(), d_durations.end(), h_durations.begin());
            thrust::copy(d_hazard.begin(), d_hazard.end(), h_hazard.begin());
            
            durations_out.insert(durations_out.end(), h_durations.begin(), h_durations.end());
            hazard_out.insert(hazard_out.end(), h_hazard.begin(), h_hazard.end());
        }
        
        // Copy Amihud results back
        std::vector<double> h_amihud(num_windows);
        thrust::copy(ctx->d_output.begin(), ctx->d_output.end(), h_amihud.begin());
        amihud_out.insert(amihud_out.end(), h_amihud.begin(), h_amihud.end());
        
        offset += gpu_n;
    }
}

void MultiGPUProcessor::process_quotes_multi_gpu(
    const std::vector<QuoteData>& quotes,
    std::vector<double>& spreads_out,
    std::vector<double>& depths_out
) {
    size_t n = quotes.size();
    if (n == 0) return;
    
    std::vector<size_t> work_sizes;
    distribute_work(n, work_sizes);
    
    spreads_out.clear();
    depths_out.clear();
    
    size_t offset = 0;
    for (int gpu = 0; gpu < num_gpus_; ++gpu) {
        cudaSetDevice(gpu);
        size_t gpu_n = work_sizes[gpu];
        
        if (gpu_n == 0) continue;
        
        // Allocate device memory
        thrust::device_vector<double> d_bid_prices(gpu_n);
        thrust::device_vector<double> d_ask_prices(gpu_n);
        thrust::device_vector<int32_t> d_bid_sizes(gpu_n);
        thrust::device_vector<int32_t> d_ask_sizes(gpu_n);
        
        // Copy data to GPU
        std::vector<double> h_bid_prices(gpu_n), h_ask_prices(gpu_n);
        std::vector<int32_t> h_bid_sizes(gpu_n), h_ask_sizes(gpu_n);
        
        for (size_t i = 0; i < gpu_n; ++i) {
            h_bid_prices[i] = quotes[offset + i].bid_price;
            h_ask_prices[i] = quotes[offset + i].ask_price;
            h_bid_sizes[i] = quotes[offset + i].bid_size;
            h_ask_sizes[i] = quotes[offset + i].ask_size;
        }
        
        thrust::copy(h_bid_prices.begin(), h_bid_prices.end(), d_bid_prices.begin());
        thrust::copy(h_ask_prices.begin(), h_ask_prices.end(), d_ask_prices.begin());
        thrust::copy(h_bid_sizes.begin(), h_bid_sizes.end(), d_bid_sizes.begin());
        thrust::copy(h_ask_sizes.begin(), h_ask_sizes.end(), d_ask_sizes.begin());
        
        // Calculate spreads
        thrust::device_vector<double> d_spreads(gpu_n);
        thrust::transform(
            d_ask_prices.begin(), d_ask_prices.end(),
            d_bid_prices.begin(),
            d_spreads.begin(),
            thrust::minus<double>()
        );
        
        // Calculate depths (simplified as bid_size + ask_size)
        thrust::device_vector<double> d_depths(gpu_n);
        thrust::transform(
            d_bid_sizes.begin(), d_bid_sizes.end(),
            d_ask_sizes.begin(),
            d_depths.begin(),
            thrust::plus<double>()
        );
        
        // Copy results back
        std::vector<double> h_spreads(gpu_n), h_depths(gpu_n);
        thrust::copy(d_spreads.begin(), d_spreads.end(), h_spreads.begin());
        thrust::copy(d_depths.begin(), d_depths.end(), h_depths.begin());
        
        spreads_out.insert(spreads_out.end(), h_spreads.begin(), h_spreads.end());
        depths_out.insert(depths_out.end(), h_depths.begin(), h_depths.end());
        
        offset += gpu_n;
    }
}

// CUDAMemoryPool implementation
CUDAMemoryPool::CUDAMemoryPool(size_t initial_size) 
    : current_size_(0), max_size_(initial_size) {
    cudaMemPoolProps pool_props = {};
    pool_props.allocType = cudaMemAllocationTypePinned;
    pool_props.handleTypes = cudaMemHandleTypePosixFileDescriptor;
    
    cudaDeviceGetDefaultMemPool(&mem_pool_, 0);
    cudaMemPoolSetAttribute(mem_pool_, cudaMemPoolAttrReleaseThreshold, &max_size_);
}

CUDAMemoryPool::~CUDAMemoryPool() {
    reset();
}

void* CUDAMemoryPool::allocate(size_t size) {
    void* ptr = nullptr;
    cudaMallocAsync(&ptr, size, 0);
    current_size_ += size;
    return ptr;
}

void CUDAMemoryPool::deallocate(void* ptr) {
    cudaFreeAsync(ptr, 0);
}

void CUDAMemoryPool::reset() {
    cudaMemPoolTrimTo(mem_pool_, 0);
    current_size_ = 0;
}

} // namespace cuda
} // namespace liquidity
