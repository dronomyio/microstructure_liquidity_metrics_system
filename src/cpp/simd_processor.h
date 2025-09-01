#pragma once

#include <immintrin.h>
#include <vector>
#include <cstdint>
#include <memory>
#include <atomic>

namespace liquidity {

struct TradeData {
    int64_t timestamp_ns;  // Nanosecond timestamp
    double price;
    double volume;
    int32_t conditions;
    char exchange;
} __attribute__((packed));

struct QuoteData {
    int64_t timestamp_ns;
    double bid_price;
    double ask_price;
    int32_t bid_size;
    int32_t ask_size;
    char bid_exchange;
    char ask_exchange;
} __attribute__((packed));

class SIMDProcessor {
public:
    SIMDProcessor();
    ~SIMDProcessor();

    // Amihud illiquidity calculation using AVX-512/AVX2
    std::vector<double> calculate_amihud_simd(
        const std::vector<TradeData>& trades,
        int window_size = 390  // Trading minutes per day
    );

    // Inter-trade duration using SIMD
    std::vector<double> calculate_durations_simd(
        const std::vector<TradeData>& trades
    );

    // Hazard rate calculation
    std::vector<double> calculate_hazard_rates_simd(
        const std::vector<double>& durations,
        double lambda = 1.0,
        double shape = 1.0
    );

    // Identify liquidity holes
    std::vector<int32_t> identify_holes_simd(
        const std::vector<double>& durations,
        double threshold_multiplier = 3.0
    );

    // Spread calculation from quotes
    std::vector<double> calculate_spreads_simd(
        const std::vector<QuoteData>& quotes
    );

private:
    // AVX-512 implementations
    void amihud_avx512(const double* prices, const double* volumes, 
                       double* output, size_t n);
    void duration_avx512(const int64_t* timestamps, double* output, size_t n);
    
    // AVX2 fallback
    void amihud_avx2(const double* prices, const double* volumes, 
                     double* output, size_t n);
    void duration_avx2(const int64_t* timestamps, double* output, size_t n);

    // Helper functions
    double median_simd(const std::vector<double>& data);
    
    bool has_avx512_;
    bool has_avx2_;
    size_t num_threads_;
};

} // namespace liquidity
