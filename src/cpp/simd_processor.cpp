// ==================== src/cpp/simd_processor.cpp ====================
#include "simd_processor.h"
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>

namespace liquidity {

SIMDProcessor::SIMDProcessor() {
    // Detect CPU capabilities
    has_avx512_ = false;
    has_avx2_ = false;
    
#ifdef __AVX512F__
    has_avx512_ = true;
#endif

#ifdef __AVX2__
    has_avx2_ = true;
#endif

    num_threads_ = omp_get_max_threads();
}

SIMDProcessor::~SIMDProcessor() {}

std::vector<double> SIMDProcessor::calculate_amihud_simd(
    const std::vector<TradeData>& trades,
    int window_size
) {
    if (trades.empty()) return {};
    
    size_t n = trades.size();
    std::vector<double> prices(n);
    std::vector<double> volumes(n);
    
    // Extract price and volume arrays
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        prices[i] = trades[i].price;
        volumes[i] = trades[i].volume;
    }
    
    // Calculate Amihud for windows
    size_t num_windows = (n + window_size - 1) / window_size;
    std::vector<double> amihud_values(num_windows);
    
    if (has_avx512_) {
        amihud_avx512(prices.data(), volumes.data(), amihud_values.data(), n);
    } else if (has_avx2_) {
        amihud_avx2(prices.data(), volumes.data(), amihud_values.data(), n);
    } else {
        // Scalar fallback
        #pragma omp parallel for
        for (size_t w = 0; w < num_windows; ++w) {
            size_t start = w * window_size;
            size_t end = std::min(start + window_size, n);
            
            if (end > start + 1) {
                double price_return = (prices[end-1] - prices[start]) / prices[start];
                double dollar_volume = 0.0;
                
                for (size_t i = start; i < end; ++i) {
                    dollar_volume += prices[i] * volumes[i];
                }
                
                amihud_values[w] = std::abs(price_return) / (dollar_volume / 1e6);
            } else {
                amihud_values[w] = 0.0;
            }
        }
    }
    
    return amihud_values;
}

std::vector<double> SIMDProcessor::calculate_durations_simd(
    const std::vector<TradeData>& trades
) {
    if (trades.size() < 2) return {};
    
    size_t n = trades.size() - 1;
    std::vector<double> durations(n);
    std::vector<int64_t> timestamps(trades.size());
    
    // Extract timestamps
    for (size_t i = 0; i < trades.size(); ++i) {
        timestamps[i] = trades[i].timestamp_ns;
    }
    
    if (has_avx512_) {
        duration_avx512(timestamps.data(), durations.data(), n);
    } else if (has_avx2_) {
        duration_avx2(timestamps.data(), durations.data(), n);
    } else {
        // Scalar fallback
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            durations[i] = static_cast<double>(timestamps[i+1] - timestamps[i]) / 1e9; // Convert to seconds
        }
    }
    
    return durations;
}

std::vector<double> SIMDProcessor::calculate_hazard_rates_simd(
    const std::vector<double>& durations,
    double lambda,
    double shape
) {
    size_t n = durations.size();
    std::vector<double> hazard_rates(n);
    
#ifdef __AVX2__
    if (has_avx2_) {
        __m256d lambda_vec = _mm256_set1_pd(lambda);
        __m256d shape_vec = _mm256_set1_pd(shape);
        __m256d shape_minus_one = _mm256_set1_pd(shape - 1.0);
        
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d t = _mm256_loadu_pd(&durations[i]);
            
            // h(t) = (shape/lambda) * (t/lambda)^(shape-1)
            __m256d t_over_lambda = _mm256_div_pd(t, lambda_vec);
            
            // Power calculation using log/exp for (t/lambda)^(shape-1)
            __m256d log_val = _mm256_log_pd(t_over_lambda);
            __m256d scaled_log = _mm256_mul_pd(log_val, shape_minus_one);
            __m256d power_result = _mm256_exp_pd(scaled_log);
            
            __m256d hazard = _mm256_mul_pd(
                _mm256_div_pd(shape_vec, lambda_vec),
                power_result
            );
            
            _mm256_storeu_pd(&hazard_rates[i], hazard);
        }
        
        // Handle remaining elements
        for (; i < n; ++i) {
            double t = durations[i];
            hazard_rates[i] = (shape / lambda) * std::pow(t / lambda, shape - 1.0);
        }
    } else
#endif
    {
        // Scalar fallback
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            double t = durations[i];
            hazard_rates[i] = (shape / lambda) * std::pow(t / lambda, shape - 1.0);
        }
    }
    
    return hazard_rates;
}

std::vector<int32_t> SIMDProcessor::identify_holes_simd(
    const std::vector<double>& durations,
    double threshold_multiplier
) {
    if (durations.empty()) return {};
    
    // Calculate median for threshold
    double median = median_simd(durations);
    double threshold = median * threshold_multiplier;
    
    size_t n = durations.size();
    std::vector<int32_t> holes(n);
    
#ifdef __AVX2__
    if (has_avx2_) {
        __m256d threshold_vec = _mm256_set1_pd(threshold);
        
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d dur = _mm256_loadu_pd(&durations[i]);
            __m256d cmp = _mm256_cmp_pd(dur, threshold_vec, _CMP_GT_OQ);
            
            // Convert comparison result to int32
            int mask = _mm256_movemask_pd(cmp);
            holes[i] = (mask & 1) ? 1 : 0;
            holes[i+1] = (mask & 2) ? 1 : 0;
            holes[i+2] = (mask & 4) ? 1 : 0;
            holes[i+3] = (mask & 8) ? 1 : 0;
        }
        
        // Handle remaining elements
        for (; i < n; ++i) {
            holes[i] = (durations[i] > threshold) ? 1 : 0;
        }
    } else
#endif
    {
        // Scalar fallback
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            holes[i] = (durations[i] > threshold) ? 1 : 0;
        }
    }
    
    return holes;
}

std::vector<double> SIMDProcessor::calculate_spreads_simd(
    const std::vector<QuoteData>& quotes
) {
    size_t n = quotes.size();
    std::vector<double> spreads(n);
    
#ifdef __AVX2__
    if (has_avx2_) {
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d ask = _mm256_set_pd(
                quotes[i+3].ask_price, quotes[i+2].ask_price,
                quotes[i+1].ask_price, quotes[i].ask_price
            );
            __m256d bid = _mm256_set_pd(
                quotes[i+3].bid_price, quotes[i+2].bid_price,
                quotes[i+1].bid_price, quotes[i].bid_price
            );
            
            __m256d spread = _mm256_sub_pd(ask, bid);
            _mm256_storeu_pd(&spreads[i], spread);
        }
        
        // Handle remaining elements
        for (; i < n; ++i) {
            spreads[i] = quotes[i].ask_price - quotes[i].bid_price;
        }
    } else
#endif
    {
        // Scalar fallback
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            spreads[i] = quotes[i].ask_price - quotes[i].bid_price;
        }
    }
    
    return spreads;
}

// AVX-512 implementations
void SIMDProcessor::amihud_avx512(
    const double* prices, 
    const double* volumes,
    double* output, 
    size_t n
) {
#ifdef __AVX512F__
    const int window_size = 390; // Default window
    size_t num_windows = (n + window_size - 1) / window_size;
    
    #pragma omp parallel for
    for (size_t w = 0; w < num_windows; ++w) {
        size_t start = w * window_size;
        size_t end = std::min(start + window_size, n);
        
        if (end <= start + 1) {
            output[w] = 0.0;
            continue;
        }
        
        // Calculate return
        double price_return = (prices[end-1] - prices[start]) / prices[start];
        
        // Calculate dollar volume using AVX-512
        __m512d sum = _mm512_setzero_pd();
        size_t i = start;
        
        for (; i + 8 <= end; i += 8) {
            __m512d p = _mm512_loadu_pd(&prices[i]);
            __m512d v = _mm512_loadu_pd(&volumes[i]);
            __m512d prod = _mm512_mul_pd(p, v);
            sum = _mm512_add_pd(sum, prod);
        }
        
        // Horizontal sum
        double dollar_volume = _mm512_reduce_add_pd(sum);
        
        // Handle remaining elements
        for (; i < end; ++i) {
            dollar_volume += prices[i] * volumes[i];
        }
        
        output[w] = std::abs(price_return) / (dollar_volume / 1e6);
    }
#endif
}

// AVX2 implementations
void SIMDProcessor::amihud_avx2(
    const double* prices,
    const double* volumes,
    double* output,
    size_t n
) {
#ifdef __AVX2__
    const int window_size = 390;
    size_t num_windows = (n + window_size - 1) / window_size;
    
    #pragma omp parallel for
    for (size_t w = 0; w < num_windows; ++w) {
        size_t start = w * window_size;
        size_t end = std::min(start + window_size, n);
        
        if (end <= start + 1) {
            output[w] = 0.0;
            continue;
        }
        
        double price_return = (prices[end-1] - prices[start]) / prices[start];
        
        __m256d sum = _mm256_setzero_pd();
        size_t i = start;
        
        for (; i + 4 <= end; i += 4) {
            __m256d p = _mm256_loadu_pd(&prices[i]);
            __m256d v = _mm256_loadu_pd(&volumes[i]);
            sum = _mm256_fmadd_pd(p, v, sum);
        }
        
        // Horizontal sum
        __m128d hi = _mm256_extractf128_pd(sum, 1);
        __m128d lo = _mm256_extractf128_pd(sum, 0);
        __m128d sum128 = _mm_add_pd(hi, lo);
        double dollar_volume = _mm_cvtsd_f64(sum128) + _mm_cvtsd_f64(_mm_shuffle_pd(sum128, sum128, 1));
        
        // Handle remaining elements
        for (; i < end; ++i) {
            dollar_volume += prices[i] * volumes[i];
        }
        
        output[w] = std::abs(price_return) / (dollar_volume / 1e6);
    }
#endif
}

void SIMDProcessor::duration_avx512(const int64_t* timestamps, double* output, size_t n) {
#ifdef __AVX512F__
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<double>(timestamps[i+1] - timestamps[i]) / 1e9;
    }
#endif
}

void SIMDProcessor::duration_avx2(const int64_t* timestamps, double* output, size_t n) {
#ifdef __AVX2__
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<double>(timestamps[i+1] - timestamps[i]) / 1e9;
    }
#endif
}

double SIMDProcessor::median_simd(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    
    size_t n = sorted.size();
    if (n % 2 == 0) {
        return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
    } else {
        return sorted[n/2];
    }
}

} // namespace liquidity
