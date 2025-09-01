// ==================== src/python/bindings.cpp ====================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../cpp/simd_processor.h"
#include "../cpp/cuda_processor.cuh"
#include "../cpp/kyle_lambda.h"
#include "../cpp/amihud_calculator.h"

namespace py = pybind11;

// Helper function to convert numpy array to vector of TradeData
std::vector<liquidity::TradeData> numpy_to_trades(py::array_t<double> arr) {
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.shape[0];
    
    std::vector<liquidity::TradeData> trades(n);
    for (size_t i = 0; i < n; ++i) {
        trades[i].timestamp_ns = static_cast<int64_t>(ptr[i * 5]);
        trades[i].price = ptr[i * 5 + 1];
        trades[i].volume = ptr[i * 5 + 2];
        trades[i].conditions = static_cast<int32_t>(ptr[i * 5 + 3]);
        trades[i].exchange = static_cast<char>(ptr[i * 5 + 4]);
    }
    
    return trades;
}

// Helper function to convert numpy array to vector of QuoteData
std::vector<liquidity::QuoteData> numpy_to_quotes(py::array_t<double> arr) {
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.shape[0];
    
    std::vector<liquidity::QuoteData> quotes(n);
    for (size_t i = 0; i < n; ++i) {
        quotes[i].timestamp_ns = static_cast<int64_t>(ptr[i * 7]);
        quotes[i].bid_price = ptr[i * 7 + 1];
        quotes[i].ask_price = ptr[i * 7 + 2];
        quotes[i].bid_size = static_cast<int32_t>(ptr[i * 7 + 3]);
        quotes[i].ask_size = static_cast<int32_t>(ptr[i * 7 + 4]);
        quotes[i].bid_exchange = static_cast<char>(ptr[i * 7 + 5]);
        quotes[i].ask_exchange = static_cast<char>(ptr[i * 7 + 6]);
    }
    
    return quotes;
}

PYBIND11_MODULE(liquidity_metrics, m) {
    m.doc() = "High-performance liquidity metrics calculation";
    
    // Initialize CUDA
    m.def("initialize_cuda", []() {
        int num_gpus;
        cudaGetDeviceCount(&num_gpus);
        return num_gpus > 0;
    });
    
    m.def("cuda_available", []() {
        int num_gpus;
        cudaGetDeviceCount(&num_gpus);
        return num_gpus > 0;
    });
    
    m.def("get_num_gpus", []() {
        int num_gpus;
        cudaGetDeviceCount(&num_gpus);
        return num_gpus;
    });
    
    m.def("get_simd_support", []() {
        std::string support = "None";
#ifdef __AVX512F__
        support = "AVX-512";
#elif defined(__AVX2__)
        support = "AVX2";
#endif
        return support;
    });
    
    // CPU calculation
    m.def("calculate_cpu", [](py::array_t<double> trades_array,
                              py::array_t<double> quotes_array,
                              std::vector<std::string> metrics) {
        auto trades = numpy_to_trades(trades_array);
        auto quotes = quotes_array.size() > 0 ? numpy_to_quotes(quotes_array) 
                                               : std::vector<liquidity::QuoteData>();
        
        py::dict results;
        liquidity::SIMDProcessor processor;
        
        for (const auto& metric : metrics) {
            if (metric == "amihud") {
                auto values = processor.calculate_amihud_simd(trades);
                results[metric.c_str()] = py::array_t<double>(values.size(), values.data());
            } else if (metric == "duration") {
                auto values = processor.calculate_durations_simd(trades);
                results[metric.c_str()] = py::array_t<double>(values.size(), values.data());
            } else if (metric == "hazard") {
                auto durations = processor.calculate_durations_simd(trades);
                auto values = processor.calculate_hazard_rates_simd(durations);
                results[metric.c_str()] = py::array_t<double>(values.size(), values.data());
            } else if (metric == "holes") {
                auto durations = processor.calculate_durations_simd(trades);
                auto values = processor.identify_holes_simd(durations);
                results[metric.c_str()] = py::array_t<int32_t>(values.size(), values.data());
            } else if (metric == "spread" && !quotes.empty()) {
                auto values = processor.calculate_spreads_simd(quotes);
                results[metric.c_str()] = py::array_t<double>(values.size(), values.data());
            }
        }
        
        return results;
    });
    
    // GPU calculation
    m.def("calculate_gpu", [](py::array_t<double> trades_array,
                              py::array_t<double> quotes_array,
                              std::vector<std::string> metrics) {
        auto trades = numpy_to_trades(trades_array);
        auto quotes = quotes_array.size() > 0 ? numpy_to_quotes(quotes_array)
                                               : std::vector<liquidity::QuoteData>();
        
        py::dict results;
        liquidity::cuda::MultiGPUProcessor processor;
        
        std::vector<double> amihud, durations, hazard;
        processor.process_trades_multi_gpu(trades, amihud, durations, hazard);
        
        for (const auto& metric : metrics) {
            if (metric == "amihud") {
                results[metric.c_str()] = py::array_t<double>(amihud.size(), amihud.data());
            } else if (metric == "duration") {
                results[metric.c_str()] = py::array_t<double>(durations.size(), durations.data());
            } else if (metric == "hazard") {
                results[metric.c_str()] = py::array_t<double>(hazard.size(), hazard.data());
            }
        }
        
        if (!quotes.empty()) {
            std::vector<double> spreads, depths;
            processor.process_quotes_multi_gpu(quotes, spreads, depths);
            
            if (std::find(metrics.begin(), metrics.end(), "spread") != metrics.end()) {
                results["spread"] = py::array_t<double>(spreads.size(), spreads.data());
            }
        }
        
        return results;
    });
}
