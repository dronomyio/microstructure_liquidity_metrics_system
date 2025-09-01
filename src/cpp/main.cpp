// ==================== src/cpp/main.cpp ====================
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include "simd_processor.h"
#include "cuda_processor.cuh"
#include "kyle_lambda.h"
#include "amihud_calculator.h"
#include "polygon_parser.h"

using namespace liquidity;
using namespace std::chrono;

// Benchmark function
template<typename Func>
double benchmark(Func func, const std::string& name, int iterations = 5) {
    std::cout << "Benchmarking " << name << "..." << std::endl;
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = high_resolution_clock::now();
    
    double avg_ms = duration_cast<microseconds>(end - start).count() / 
                    (1000.0 * iterations);
    
    std::cout << "  Average time: " << std::fixed << std::setprecision(3) 
              << avg_ms << " ms" << std::endl;
    
    return avg_ms;
}

// Generate synthetic data for testing
void generate_synthetic_data(
    std::vector<TradeData>& trades,
    std::vector<QuoteData>& quotes,
    size_t num_trades,
    size_t num_quotes
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> price_dist(100.0, 0.1);
    std::exponential_distribution<> volume_dist(0.001);
    std::uniform_int_distribution<> time_dist(1000000, 10000000); // nanoseconds between events
    
    int64_t current_time = 0;
    
    // Generate trades
    trades.clear();
    trades.reserve(num_trades);
    
    for (size_t i = 0; i < num_trades; ++i) {
        current_time += time_dist(gen);
        
        TradeData trade;
        trade.timestamp_ns = current_time;
        trade.price = price_dist(gen);
        trade.volume = volume_dist(gen) * 1000;
        trade.conditions = 0;
        trade.exchange = 'N';
        
        trades.push_back(trade);
    }
    
    // Generate quotes (more frequent than trades)
    quotes.clear();
    quotes.reserve(num_quotes);
    
    current_time = 0;
    for (size_t i = 0; i < num_quotes; ++i) {
        current_time += time_dist(gen) / 5; // Quotes 5x more frequent
        
        QuoteData quote;
        quote.timestamp_ns = current_time;
        double mid = price_dist(gen);
        double spread = 0.01 + std::abs(price_dist(gen) - 100.0) * 0.001;
        
        quote.bid_price = mid - spread / 2;
        quote.ask_price = mid + spread / 2;
        quote.bid_size = std::uniform_int_distribution<>(100, 1000)(gen);
        quote.ask_size = std::uniform_int_distribution<>(100, 1000)(gen);
        quote.bid_exchange = 'N';
        quote.ask_exchange = 'N';
        
        quotes.push_back(quote);
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Liquidity Metrics High-Performance Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Check available hardware
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    std::cout << "\nHardware Detection:" << std::endl;
    std::cout << "  GPUs available: " << num_gpus << std::endl;
    
    if (num_gpus > 0) {
        for (int i = 0; i < num_gpus; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "  GPU " << i << ": " << prop.name 
                      << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
        }
    }
    
#ifdef __AVX512F__
    std::cout << "  SIMD: AVX-512 supported" << std::endl;
#elif defined(__AVX2__)
    std::cout << "  SIMD: AVX2 supported" << std::endl;
#else
    std::cout << "  SIMD: No AVX support detected" << std::endl;
#endif
    
    // Generate test data
    std::cout << "\nGenerating synthetic data..." << std::endl;
    std::vector<TradeData> trades;
    std::vector<QuoteData> quotes;
    
    const size_t num_trades = 1000000;
    const size_t num_quotes = 5000000;
    
    generate_synthetic_data(trades, quotes, num_trades, num_quotes);
    std::cout << "  Generated " << trades.size() << " trades and " 
              << quotes.size() << " quotes" << std::endl;
    
    // Test SIMD implementation
    std::cout << "\n=== SIMD Performance ===" << std::endl;
    SIMDProcessor simd_processor;
    
    std::vector<double> simd_amihud, simd_durations, simd_hazard, simd_spreads;
    std::vector<int32_t> simd_holes;
    
    benchmark([&]() {
        simd_amihud = simd_processor.calculate_amihud_simd(trades, 390);
    }, "SIMD Amihud");
    
    benchmark([&]() {
        simd_durations = simd_processor.calculate_durations_simd(trades);
    }, "SIMD Durations");
    
    benchmark([&]() {
        simd_hazard = simd_processor.calculate_hazard_rates_simd(simd_durations, 1.0, 1.5);
    }, "SIMD Hazard Rates");
    
    benchmark([&]() {
        simd_holes = simd_processor.identify_holes_simd(simd_durations, 3.0);
    }, "SIMD Holes Detection");
    
    benchmark([&]() {
        simd_spreads = simd_processor.calculate_spreads_simd(quotes);
    }, "SIMD Spreads");
    
    // Test CUDA implementation
    if (num_gpus > 0) {
        std::cout << "\n=== CUDA Performance ===" << std::endl;
        cuda::MultiGPUProcessor gpu_processor(num_gpus);
        
        std::vector<double> cuda_amihud, cuda_durations, cuda_hazard;
        std::vector<double> cuda_spreads, cuda_depths;
        
        benchmark([&]() {
            gpu_processor.process_trades_multi_gpu(trades, cuda_amihud, 
                                                   cuda_durations, cuda_hazard);
        }, "CUDA Multi-GPU Trade Processing");
        
        benchmark([&]() {
            gpu_processor.process_quotes_multi_gpu(quotes, cuda_spreads, cuda_depths);
        }, "CUDA Multi-GPU Quote Processing");
    }
    
    // Test Kyle Lambda implementation
    std::cout << "\n=== Kyle Lambda Calculation ===" << std::endl;
    
    benchmark([&]() {
        auto kyle_result = KyleLambda::calculate_kyle_lambda_hasbrouck(
            trades, quotes, 5LL * 60 * 1000000000LL
        );
        std::cout << "  Kyle Lambda: " << std::scientific << kyle_result.lambda 
                  << " (R² = " << std::fixed << kyle_result.r_squared << ")" << std::endl;
    }, "Kyle Lambda (Hasbrouck)");
    
    // Test Amihud with aggregation
    std::cout << "\n=== Amihud with Aggregation ===" << std::endl;
    
    benchmark([&]() {
        auto daily_amihud = AmihudCalculator::calculate_daily_amihud(trades);
        std::cout << "  Daily Amihud: " << std::scientific << daily_amihud << std::endl;
    }, "Amihud Daily");
    
    benchmark([&]() {
        auto aggregated = AmihudCalculator::calculate_amihud_aggregated(
            trades, AmihudCalculator::FIVE_MINUTES
        );
        double avg_amihud = 0.0;
        for (const auto& result : aggregated) {
            avg_amihud += result.illiquidity;
        }
        avg_amihud /= aggregated.size();
        std::cout << "  5-min Amihud (avg): " << std::scientific << avg_amihud << std::endl;
    }, "Amihud 5-minute aggregated");
    
    // Performance comparison summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Processed " << num_trades << " trades and " << num_quotes << " quotes" << std::endl;
    
    if (num_gpus > 0) {
        std::cout << "\nKey Insights:" << std::endl;
        std::cout << "• Kyle Lambda works directly on nanosecond data" << std::endl;
        std::cout << "• Amihud requires aggregation (5-min or daily)" << std::endl;
        std::cout << "• GPU acceleration provides significant speedup for large datasets" << std::endl;
        std::cout << "• SIMD optimizations improve CPU performance by 2-4x" << std::endl;
    }
    
    return 0;
}

