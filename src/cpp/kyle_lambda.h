// src/cpp/kyle_lambda.h
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include "polygon_parser.h"

namespace liquidity {

/**
 * Kyle (1985) Lambda Implementation
 * 
 * Lambda measures the permanent price impact per unit of net order flow.
 * In Kyle's model: ΔP = λ * Q
 * Where:
 *   ΔP = permanent price change
 *   λ = price impact coefficient (market depth inverse)
 *   Q = net order flow (signed volume)
 */
class KyleLambda {
public:
    struct LambdaResult {
        double lambda;              // Price impact per unit volume
        double temporary_impact;    // Temporary component (bid-ask bounce)
        double permanent_impact;    // Permanent component (information)
        double r_squared;          // Regression R²
        int64_t timestamp_ns;
        int num_trades;
    };

    struct TradeImpact {
        int64_t timestamp_ns;
        double pre_midpoint;       // Midpoint before trade
        double post_midpoint;      // Midpoint after trade
        double trade_price;
        double signed_volume;      // Positive for buy, negative for sell
        double price_impact;       // post_midpoint - pre_midpoint
        bool is_buy;              // Trade direction
    };

    /**
     * Calculate Kyle's lambda using the Hasbrouck (1991) regression approach
     * This is the most robust method for real data
     */
    static LambdaResult calculate_kyle_lambda_hasbrouck(
        const std::vector<TradeData>& trades,
        const std::vector<QuoteData>& quotes,
        int64_t window_ns = 5LL * 60 * 1000000000LL  // 5-minute windows
    ) {
        if (trades.empty() || quotes.empty()) {
            return {0.0, 0.0, 0.0, 0.0, 0, 0};
        }

        std::vector<TradeImpact> impacts;
        
        // Match trades with quotes to get midpoints
        size_t quote_idx = 0;
        
        for (size_t i = 0; i < trades.size(); ++i) {
            const auto& trade = trades[i];
            
            // Find quote immediately before trade
            while (quote_idx < quotes.size() - 1 && 
                   quotes[quote_idx + 1].timestamp_ns < trade.timestamp_ns) {
                quote_idx++;
            }
            
            if (quote_idx >= quotes.size() - 1) break;
            
            // Get pre-trade midpoint
            double pre_midpoint = (quotes[quote_idx].bid_price + 
                                  quotes[quote_idx].ask_price) / 2.0;
            
            // Find quote immediately after trade (within 1 second)
            size_t post_quote_idx = quote_idx;
            while (post_quote_idx < quotes.size() - 1 && 
                   quotes[post_quote_idx].timestamp_ns < 
                   trade.timestamp_ns + 1000000000LL) {  // 1 second later
                post_quote_idx++;
            }
            
            if (post_quote_idx >= quotes.size()) continue;
            
            double post_midpoint = (quotes[post_quote_idx].bid_price + 
                                   quotes[post_quote_idx].ask_price) / 2.0;
            
            // Determine trade direction (Lee-Ready algorithm)
            bool is_buy = determine_trade_direction(
                trade, quotes[quote_idx], pre_midpoint
            );
            
            double signed_volume = is_buy ? trade.volume : -trade.volume;
            double price_impact = post_midpoint - pre_midpoint;
            
            impacts.push_back({
                trade.timestamp_ns,
                pre_midpoint,
                post_midpoint,
                trade.price,
                signed_volume,
                price_impact,
                is_buy
            });
        }
        
        // Run regression: price_impact = λ * signed_volume + ε
        return run_kyle_regression(impacts);
    }

    /**
     * Calculate Kyle's lambda using the original Kyle (1985) model
     * Best for theoretical analysis or when order flow is known
     */
    static LambdaResult calculate_kyle_lambda_original(
        const std::vector<TradeData>& trades,
        double total_variance,  // Total return variance
        double noise_variance   // Variance of noise trader volume
    ) {
        // In Kyle's original model:
        // λ = σ_v / (2 * σ_u)
        // where σ_v = standard deviation of asset value
        //       σ_u = standard deviation of noise trader demand
        
        double sigma_v = std::sqrt(total_variance);
        double sigma_u = std::sqrt(noise_variance);
        
        double lambda = sigma_v / (2.0 * sigma_u);
        
        return {
            lambda,
            0.0,  // No temporary impact in original model
            lambda,  // All impact is permanent
            1.0,  // Perfect model fit by construction
            trades.empty() ? 0 : trades[0].timestamp_ns,
            static_cast<int>(trades.size())
        };
    }

    /**
     * Calculate realized price impact (for validation)
     * This directly measures the actual impact of trades
     */
    static std::vector<double> calculate_realized_impact(
        const std::vector<TradeData>& trades,
        const std::vector<QuoteData>& quotes,
        int64_t impact_horizon_ns = 5LL * 60 * 1000000000LL  // 5 minutes
    ) {
        std::vector<double> realized_impacts;
        
        for (size_t i = 0; i < trades.size(); ++i) {
            const auto& trade = trades[i];
            
            // Find midpoint at trade time
            auto quote_at_trade = find_quote_at_time(quotes, trade.timestamp_ns);
            double initial_midpoint = (quote_at_trade.bid_price + 
                                       quote_at_trade.ask_price) / 2.0;
            
            // Find midpoint after impact horizon
            auto quote_after = find_quote_at_time(
                quotes, trade.timestamp_ns + impact_horizon_ns
            );
            double final_midpoint = (quote_after.bid_price + 
                                     quote_after.ask_price) / 2.0;
            
            // Calculate price impact per unit volume
            double impact_per_volume = (final_midpoint - initial_midpoint) / 
                                       (trade.price * trade.volume / 1e6);  // Per million $
            
            realized_impacts.push_back(impact_per_volume);
        }
        
        return realized_impacts;
    }

    /**
     * SIMD-optimized batch calculation of Kyle's lambda
     */
    static void calculate_kyle_lambda_simd(
        const float* signed_volumes,
        const float* price_impacts,
        float* lambdas,
        int num_windows,
        int trades_per_window
    ) {
        #ifdef __AVX2__
        for (int w = 0; w < num_windows; w++) {
            int offset = w * trades_per_window;
            
            // Calculate components for regression using SIMD
            __m256 sum_xy = _mm256_setzero_ps();
            __m256 sum_x2 = _mm256_setzero_ps();
            __m256 sum_x = _mm256_setzero_ps();
            __m256 sum_y = _mm256_setzero_ps();
            
            for (int i = 0; i < trades_per_window; i += 8) {
                __m256 x = _mm256_loadu_ps(&signed_volumes[offset + i]);
                __m256 y = _mm256_loadu_ps(&price_impacts[offset + i]);
                
                sum_xy = _mm256_fmadd_ps(x, y, sum_xy);
                sum_x2 = _mm256_fmadd_ps(x, x, sum_x2);
                sum_x = _mm256_add_ps(sum_x, x);
                sum_y = _mm256_add_ps(sum_y, y);
            }
            
            // Horizontal sum
            float h_sum_xy = horizontal_sum_avx2(sum_xy);
            float h_sum_x2 = horizontal_sum_avx2(sum_x2);
            float h_sum_x = horizontal_sum_avx2(sum_x);
            float h_sum_y = horizontal_sum_avx2(sum_y);
            
            // Calculate lambda (slope of regression)
            float n = trades_per_window;
            float lambda = (n * h_sum_xy - h_sum_x * h_sum_y) / 
                          (n * h_sum_x2 - h_sum_x * h_sum_x);
            
            lambdas[w] = lambda;
        }
        #else
        // Scalar fallback
        for (int w = 0; w < num_windows; w++) {
            lambdas[w] = calculate_lambda_scalar(
                &signed_volumes[w * trades_per_window],
                &price_impacts[w * trades_per_window],
                trades_per_window
            );
        }
        #endif
    }

private:
    /**
     * Determine trade direction using Lee-Ready algorithm
     */
    static bool determine_trade_direction(
        const TradeData& trade,
        const QuoteData& quote,
        double midpoint
    ) {
        // Lee-Ready algorithm:
        // 1. If trade price > midpoint, it's a buy
        // 2. If trade price < midpoint, it's a sell
        // 3. If trade price = midpoint, use tick test
        
        if (trade.price > midpoint + 1e-8) {
            return true;  // Buy
        } else if (trade.price < midpoint - 1e-8) {
            return false;  // Sell
        } else {
            // Tick test: compare with previous trade
            // For simplicity, assuming buy if at ask, sell if at bid
            double ask_distance = std::abs(trade.price - quote.ask_price);
            double bid_distance = std::abs(trade.price - quote.bid_price);
            return ask_distance < bid_distance;
        }
    }

    /**
     * Run regression to estimate Kyle's lambda
     */
    static LambdaResult run_kyle_regression(
        const std::vector<TradeImpact>& impacts
    ) {
        if (impacts.size() < 2) {
            return {0.0, 0.0, 0.0, 0.0, 0, 0};
        }
        
        // Simple OLS regression: price_impact = λ * signed_volume
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
        int n = impacts.size();
        
        for (const auto& impact : impacts) {
            sum_x += impact.signed_volume;
            sum_y += impact.price_impact;
            sum_xy += impact.signed_volume * impact.price_impact;
            sum_x2 += impact.signed_volume * impact.signed_volume;
        }
        
        // Calculate lambda (slope)
        double lambda = (n * sum_xy - sum_x * sum_y) / 
                       (n * sum_x2 - sum_x * sum_x);
        
        // Calculate R-squared
        double mean_y = sum_y / n;
        double ss_tot = 0.0, ss_res = 0.0;
        
        for (const auto& impact : impacts) {
            double predicted = lambda * impact.signed_volume;
            ss_res += std::pow(impact.price_impact - predicted, 2);
            ss_tot += std::pow(impact.price_impact - mean_y, 2);
        }
        
        double r_squared = 1.0 - (ss_res / ss_tot);
        
        // Decompose into temporary and permanent (simplified)
        // Temporary impact is typically the bid-ask bounce component
        double avg_spread = calculate_average_spread(impacts);
        double temporary = avg_spread / 2.0;  // Half-spread
        double permanent = lambda;
        
        return {
            lambda,
            temporary,
            permanent,
            r_squared,
            impacts[0].timestamp_ns,
            n
        };
    }

    static double calculate_average_spread(const std::vector<TradeImpact>& impacts) {
        double sum = 0.0;
        for (const auto& impact : impacts) {
            sum += 2.0 * std::abs(impact.trade_price - 
                                 (impact.pre_midpoint + impact.post_midpoint) / 2.0);
        }
        return sum / impacts.size();
    }

    static QuoteData find_quote_at_time(
        const std::vector<QuoteData>& quotes,
        int64_t timestamp_ns
    ) {
        // Binary search for efficiency
        auto it = std::lower_bound(
            quotes.begin(), quotes.end(), timestamp_ns,
            [](const QuoteData& q, int64_t t) { return q.timestamp_ns < t; }
        );
        
        if (it != quotes.begin()) --it;
        return *it;
    }

    static float horizontal_sum_avx2(__m256 v) {
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 lo = _mm256_extractf128_ps(v, 0);
        lo = _mm_add_ps(hi, lo);
        hi = _mm_movehl_ps(hi, lo);
        lo = _mm_add_ps(hi, lo);
        hi = _mm_shuffle_ps(lo, lo, 1);
        lo = _mm_add_ss(hi, lo);
        return _mm_cvtss_f32(lo);
    }

    static float calculate_lambda_scalar(
        const float* signed_volumes,
        const float* price_impacts,
        int n
    ) {
        float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        for (int i = 0; i < n; i++) {
            sum_x += signed_volumes[i];
            sum_y += price_impacts[i];
            sum_xy += signed_volumes[i] * price_impacts[i];
            sum_x2 += signed_volumes[i] * signed_volumes[i];
        }
        return (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    }
};

} // namespace liquidity
