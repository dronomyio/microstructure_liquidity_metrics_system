// src/cpp/kyle_lambda.cu
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace liquidity {
namespace cuda {

/**
 * CUDA kernel for matching trades with quotes and calculating price impacts
 */
__global__ void calculate_trade_impacts_kernel(
    const int64_t* __restrict__ trade_timestamps,
    const double* __restrict__ trade_prices,
    const double* __restrict__ trade_volumes,
    const int64_t* __restrict__ quote_timestamps,
    const double* __restrict__ bid_prices,
    const double* __restrict__ ask_prices,
    double* __restrict__ signed_volumes,
    double* __restrict__ price_impacts,
    const int num_trades,
    const int num_quotes,
    const int64_t impact_horizon_ns
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_trades) return;
    
    int64_t trade_time = trade_timestamps[tid];
    double trade_price = trade_prices[tid];
    double trade_volume = trade_volumes[tid];
    
    // Binary search for quote before trade (simplified linear search here)
    int pre_quote_idx = 0;
    for (int i = 0; i < num_quotes - 1; i++) {
        if (quote_timestamps[i + 1] > trade_time) {
            pre_quote_idx = i;
            break;
        }
    }
    
    // Get pre-trade midpoint
    double pre_midpoint = (bid_prices[pre_quote_idx] + 
                          ask_prices[pre_quote_idx]) / 2.0;
    
    // Find quote after impact horizon
    int post_quote_idx = pre_quote_idx;
    int64_t target_time = trade_time + impact_horizon_ns;
    for (int i = pre_quote_idx; i < num_quotes; i++) {
        if (quote_timestamps[i] >= target_time) {
            post_quote_idx = i;
            break;
        }
    }
    
    // Get post-trade midpoint
    double post_midpoint = (bid_prices[post_quote_idx] + 
                           ask_prices[post_quote_idx]) / 2.0;
    
    // Determine trade direction (Lee-Ready algorithm)
    bool is_buy = trade_price > pre_midpoint;
    if (abs(trade_price - pre_midpoint) < 1e-8) {
        // At midpoint - use quote test
        double ask_dist = abs(trade_price - ask_prices[pre_quote_idx]);
        double bid_dist = abs(trade_price - bid_prices[pre_quote_idx]);
        is_buy = ask_dist < bid_dist;
    }
    
    // Calculate signed volume and price impact
    signed_volumes[tid] = is_buy ? trade_volume : -trade_volume;
    price_impacts[tid] = post_midpoint - pre_midpoint;
}

/**
 * CUDA kernel for Kyle regression using shared memory
 */
__global__ void kyle_regression_kernel(
    const double* __restrict__ signed_volumes,
    const double* __restrict__ price_impacts,
    double* __restrict__ lambdas,
    double* __restrict__ r_squared_values,
    const int num_windows,
    const int trades_per_window
) {
    extern __shared__ double shared_data[];
    
    int window_id = blockIdx.x;
    if (window_id >= num_windows) return;
    
    int tid = threadIdx.x;
    int offset = window_id * trades_per_window;
    
    // Divide shared memory
    double* s_sum_x = &shared_data[0];
    double* s_sum_y = &shared_data[blockDim.x];
    double* s_sum_xy = &shared_data[2 * blockDim.x];
    double* s_sum_x2 = &shared_data[3 * blockDim.x];
    
    // Initialize sums
    s_sum_x[tid] = 0.0;
    s_sum_y[tid] = 0.0;
    s_sum_xy[tid] = 0.0;
    s_sum_x2[tid] = 0.0;
    
    // Each thread processes multiple trades
    for (int i = tid; i < trades_per_window; i += blockDim.x) {
        if (offset + i < num_windows * trades_per_window) {
            double x = signed_volumes[offset + i];
            double y = price_impacts[offset + i];
            
            s_sum_x[tid] += x;
            s_sum_y[tid] += y;
            s_sum_xy[tid] += x * y;
            s_sum_x2[tid] += x * x;
        }
    }
    
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum_x[tid] += s_sum_x[tid + stride];
            s_sum_y[tid] += s_sum_y[tid + stride];
            s_sum_xy[tid] += s_sum_xy[tid + stride];
            s_sum_x2[tid] += s_sum_x2[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 calculates final lambda and R²
    if (tid == 0) {
        double n = trades_per_window;
        double sum_x = s_sum_x[0];
        double sum_y = s_sum_y[0];
        double sum_xy = s_sum_xy[0];
        double sum_x2 = s_sum_x2[0];
        
        // Calculate lambda (slope)
        double denominator = n * sum_x2 - sum_x * sum_x;
        double lambda = (denominator != 0.0) ? 
                       (n * sum_xy - sum_x * sum_y) / denominator : 0.0;
        
        lambdas[window_id] = lambda;
        
        // Calculate R² for quality check
        double mean_y = sum_y / n;
        double ss_tot = 0.0;
        double ss_res = 0.0;
        
        // Second pass for R² (simplified)
        for (int i = 0; i < trades_per_window; i++) {
            double x = signed_volumes[offset + i];
            double y = price_impacts[offset + i];
            double y_pred = lambda * x;
            
            ss_res += (y - y_pred) * (y - y_pred);
            ss_tot += (y - mean_y) * (y - mean_y);
        }
        
        r_squared_values[window_id] = (ss_tot != 0.0) ? 
                                      1.0 - ss_res / ss_tot : 0.0;
    }
}

/**
 * Advanced Kyle lambda estimation using cuSolver for robust regression
 */
class KyleLambdaGPU {
public:
    struct Config {
        int64_t impact_horizon_ns = 5LL * 60 * 1000000000LL;  // 5 minutes
        int window_size = 100;  // Trades per regression window
        bool use_robust_regression = false;
        int num_gpus = 1;
    };
    
    static void calculate_kyle_lambda_multi_gpu(
        const thrust::device_vector<int64_t>& d_trade_timestamps,
        const thrust::device_vector<double>& d_trade_prices,
        const thrust::device_vector<double>& d_trade_volumes,
        const thrust::device_vector<int64_t>& d_quote_timestamps,
        const thrust::device_vector<double>& d_bid_prices,
        const thrust::device_vector<double>& d_ask_prices,
        thrust::device_vector<double>& d_lambdas,
        const Config& config = Config()
    ) {
        int num_trades = d_trade_timestamps.size();
        int num_quotes = d_quote_timestamps.size();
        
        // Allocate intermediate arrays
        thrust::device_vector<double> d_signed_volumes(num_trades);
        thrust::device_vector<double> d_price_impacts(num_trades);
        
        // Calculate trade impacts
        int block_size = 256;
        int grid_size = (num_trades + block_size - 1) / block_size;
        
        calculate_trade_impacts_kernel<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(d_trade_timestamps.data()),
            thrust::raw_pointer_cast(d_trade_prices.data()),
            thrust::raw_pointer_cast(d_trade_volumes.data()),
            thrust::raw_pointer_cast(d_quote_timestamps.data()),
            thrust::raw_pointer_cast(d_bid_prices.data()),
            thrust::raw_pointer_cast(d_ask_prices.data()),
            thrust::raw_pointer_cast(d_signed_volumes.data()),
            thrust::raw_pointer_cast(d_price_impacts.data()),
            num_trades,
            num_quotes,
            config.impact_horizon_ns
        );
        
        cudaDeviceSynchronize();
        
        // Calculate Kyle lambda for windows
        int num_windows = (num_trades + config.window_size - 1) / config.window_size;
        d_lambdas.resize(num_windows);
        thrust::device_vector<double> d_r_squared(num_windows);
        
        // Use shared memory for reduction
        size_t shared_mem_size = 4 * block_size * sizeof(double);
        
        kyle_regression_kernel<<<num_windows, block_size, shared_mem_size>>>(
            thrust::raw_pointer_cast(d_signed_volumes.data()),
            thrust::raw_pointer_cast(d_price_impacts.data()),
            thrust::raw_pointer_cast(d_lambdas.data()),
            thrust::raw_pointer_cast(d_r_squared.data()),
            num_windows,
            config.window_size
        );
        
        cudaDeviceSynchronize();
        
        // Optional: Use cuSolver for robust regression
        if (config.use_robust_regression) {
            calculate_robust_kyle_lambda(
                d_signed_volumes,
                d_price_impacts,
                d_lambdas,
                config
            );
        }
    }
    
private:
    /**
     * Robust regression using cuSolver (handles outliers better)
     */
    static void calculate_robust_kyle_lambda(
        const thrust::device_vector<double>& d_signed_volumes,
        const thrust::device_vector<double>& d_price_impacts,
        thrust::device_vector<double>& d_lambdas,
        const Config& config
    ) {
        cusolverDnHandle_t cusolver_handle;
        cusolverDnCreate(&cusolver_handle);
        
        int n = d_signed_volumes.size();
        int num_windows = d_lambdas.size();
        
        for (int w = 0; w < num_windows; w++) {
            int start_idx = w * config.window_size;
            int end_idx = std::min(start_idx + config.window_size, n);
            int window_n = end_idx - start_idx;
            
            // Create design matrix X (window_n x 2) with intercept
            thrust::device_vector<double> d_X(window_n * 2);
            thrust::device_vector<double> d_y(window_n);
            
            // Fill design matrix
            thrust::transform(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(window_n),
                d_X.begin(),
                [=] __device__ (int i) { return 1.0; }  // Intercept column
            );
            
            thrust::copy(
                d_signed_volumes.begin() + start_idx,
                d_signed_volumes.begin() + end_idx,
                d_X.begin() + window_n  // Second column
            );
            
            thrust::copy(
                d_price_impacts.begin() + start_idx,
                d_price_impacts.begin() + end_idx,
                d_y.begin()
            );
            
            // Solve least squares problem
            thrust::device_vector<double> d_coeffs(2);
            int work_size;
            
            cusolverDnDgels(
                cusolver_handle,
                CUBLAS_OP_N,
                window_n, 2, 1,
                thrust::raw_pointer_cast(d_X.data()), window_n,
                thrust::raw_pointer_cast(d_y.data()), window_n,
                thrust::raw_pointer_cast(d_coeffs.data()),
                &work_size,
                nullptr
            );
            
            // Lambda is the slope (second coefficient)
            d_lambdas[w] = d_coeffs[1];
        }
        
        cusolverDnDestroy(cusolver_handle);
    }
};

/**
 * Real-time Kyle lambda calculation kernel for streaming data
 */
__global__ void streaming_kyle_lambda_kernel(
    const double* __restrict__ new_signed_volume,
    const double* __restrict__ new_price_impact,
    double* __restrict__ running_sum_x,
    double* __restrict__ running_sum_y,
    double* __restrict__ running_sum_xy,
    double* __restrict__ running_sum_x2,
    int* __restrict__ running_count,
    double* __restrict__ current_lambda,
    const int batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    // Update running sums atomically
    atomicAdd(running_sum_x, new_signed_volume[tid]);
    atomicAdd(running_sum_y, new_price_impact[tid]);
    atomicAdd(running_sum_xy, new_signed_volume[tid] * new_price_impact[tid]);
    atomicAdd(running_sum_x2, new_signed_volume[tid] * new_signed_volume[tid]);
    atomicAdd(running_count, 1);
    
    // Thread 0 updates lambda
    if (tid == 0) {
        __threadfence();  // Ensure all updates are visible
        
        double n = *running_count;
        double sum_x = *running_sum_x;
        double sum_y = *running_sum_y;
        double sum_xy = *running_sum_xy;
        double sum_x2 = *running_sum_x2;
        
        double denominator = n * sum_x2 - sum_x * sum_x;
        if (denominator != 0.0) {
            *current_lambda = (n * sum_xy - sum_x * sum_y) / denominator;
        }
    }
}

} // namespace cuda
} // namespace liquidity
