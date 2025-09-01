# src/python/kyle_lambda_metrics.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
import warnings

@dataclass
class KyleLambdaResult:
    """Results from Kyle lambda estimation"""
    lambda_value: float
    temporary_impact: float
    permanent_impact: float
    r_squared: float
    num_trades: int
    timestamp_ns: int
    
    @property
    def total_impact(self) -> float:
        return self.temporary_impact + self.permanent_impact

class KyleLambdaCalculator:
    """
    Implementation of Kyle (1985) lambda for nanosecond-precision data.
    
    Kyle's lambda measures the price impact per unit of net order flow,
    which is ideal for high-frequency data as it works at the trade level
    without requiring aggregation.
    """
    
    @staticmethod
    def determine_trade_direction(
        trades_df: pd.DataFrame,
        quotes_df: pd.DataFrame
    ) -> pd.Series:
        """
        Implement Lee-Ready algorithm to determine trade direction.
        
        Returns:
            Series of bool values (True = buy, False = sell)
        """
        # Merge trades with quotes (asof merge for nearest quote before trade)
        trades_df = trades_df.sort_values('timestamp_ns')
        quotes_df = quotes_df.sort_values('timestamp_ns')
        
        merged = pd.merge_asof(
            trades_df,
            quotes_df[['timestamp_ns', 'bid_price', 'ask_price']],
            on='timestamp_ns',
            direction='backward',
            suffixes=('', '_quote')
        )
        
        # Calculate midpoint
        merged['midpoint'] = (merged['bid_price'] + merged['ask_price']) / 2
        
        # Lee-Ready algorithm
        # 1. Trade above midpoint = buy
        # 2. Trade below midpoint = sell  
        # 3. Trade at midpoint = use tick test
        
        conditions = [
            merged['price'] > merged['midpoint'] + 1e-8,
            merged['price'] < merged['midpoint'] - 1e-8
        ]
        choices = [True, False]
        
        # Default to tick test for trades at midpoint
        merged['is_buy'] = np.select(
            conditions, 
            choices,
            default=(merged['price'].diff() >= 0)  # Uptick = buy
        )
        
        return merged['is_buy']
    
    @staticmethod
    def calculate_kyle_lambda(
        trades_df: pd.DataFrame,
        quotes_df: pd.DataFrame,
        impact_horizon_ns: int = 5 * 60 * 1_000_000_000  # 5 minutes
    ) -> KyleLambdaResult:
        """
        Calculate Kyle's lambda using Hasbrouck (1991) regression approach.
        
        Args:
            trades_df: DataFrame with columns ['timestamp_ns', 'price', 'volume']
            quotes_df: DataFrame with columns ['timestamp_ns', 'bid_price', 'ask_price']
            impact_horizon_ns: Time horizon for measuring price impact
            
        Returns:
            KyleLambdaResult with estimated parameters
        """
        if len(trades_df) < 10:
            warnings.warn("Too few trades for reliable lambda estimation")
            return KyleLambdaResult(0, 0, 0, 0, len(trades_df), 0)
        
        # Determine trade directions
        trades_df['is_buy'] = KyleLambdaCalculator.determine_trade_direction(
            trades_df, quotes_df
        )
        
        # Calculate signed volume (in millions for scaling)
        trades_df['signed_volume'] = np.where(
            trades_df['is_buy'],
            trades_df['volume'] * trades_df['price'] / 1e6,  # Buy volume
            -trades_df['volume'] * trades_df['price'] / 1e6  # Sell volume
        )
        
        # Calculate price impacts
        impacts = []
        
        for idx, trade in trades_df.iterrows():
            # Get midpoint before trade
            pre_quotes = quotes_df[quotes_df['timestamp_ns'] < trade['timestamp_ns']]
            if len(pre_quotes) == 0:
                continue
                
            pre_midpoint = (pre_quotes.iloc[-1]['bid_price'] + 
                           pre_quotes.iloc[-1]['ask_price']) / 2
            
            # Get midpoint after impact horizon
            target_time = trade['timestamp_ns'] + impact_horizon_ns
            post_quotes = quotes_df[quotes_df['timestamp_ns'] <= target_time]
            
            if len(post_quotes) == 0:
                continue
                
            post_midpoint = (post_quotes.iloc[-1]['bid_price'] + 
                            post_quotes.iloc[-1]['ask_price']) / 2
            
            price_impact = post_midpoint - pre_midpoint
            
            impacts.append({
                'signed_volume': trade['signed_volume'],
                'price_impact': price_impact,
                'timestamp_ns': trade['timestamp_ns']
            })
        
        if len(impacts) < 2:
            return KyleLambdaResult(0, 0, 0, 0, len(trades_df), 0)
        
        impacts_df = pd.DataFrame(impacts)
        
        # Run regression: price_impact = α + λ * signed_volume
        X = impacts_df['signed_volume'].values
        y = impacts_df['price_impact'].values
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # OLS regression
        coeffs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        
        intercept = coeffs[0]
        lambda_value = coeffs[1]
        
        # Calculate R²
        y_pred = X_with_intercept @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Decompose impact (simplified)
        # Temporary impact ≈ half-spread component
        avg_spread = (quotes_df['ask_price'] - quotes_df['bid_price']).mean()
        temporary_impact = avg_spread / 2
        permanent_impact = abs(lambda_value)
        
        return KyleLambdaResult(
            lambda_value=lambda_value,
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            r_squared=r_squared,
            num_trades=len(impacts),
            timestamp_ns=trades_df.iloc[0]['timestamp_ns']
        )
    
    @staticmethod
    def calculate_intraday_kyle_lambda(
        trades_df: pd.DataFrame,
        quotes_df: pd.DataFrame,
        window_size: int = 100,  # Number of trades per window
        min_trades: int = 30
    ) -> pd.DataFrame:
        """
        Calculate Kyle's lambda for rolling windows throughout the day.
        
        This shows how market depth changes over time.
        """
        results = []
        
        # Sort by timestamp
        trades_df = trades_df.sort_values('timestamp_ns')
        
        # Calculate lambda for each window
        for i in range(0, len(trades_df) - min_trades, window_size // 2):  # 50% overlap
            window_trades = trades_df.iloc[i:i + window_size]
            
            if len(window_trades) < min_trades:
                continue
            
            # Find relevant quotes for this time window
            start_time = window_trades.iloc[0]['timestamp_ns']
            end_time = window_trades.iloc[-1]['timestamp_ns']
            window_quotes = quotes_df[
                (quotes_df['timestamp_ns'] >= start_time - 1_000_000_000) &  # 1 sec before
                (quotes_df['timestamp_ns'] <= end_time + 5 * 60 * 1_000_000_000)  # 5 min after
            ]
            
            if len(window_quotes) < 10:
                continue
            
            result = KyleLambdaCalculator.calculate_kyle_lambda(
                window_trades, 
                window_quotes,
                impact_horizon_ns=60 * 1_000_000_000  # 1 minute for intraday
            )
            
            results.append({
                'timestamp_ns': window_trades.iloc[len(window_trades)//2]['timestamp_ns'],
                'lambda': result.lambda_value,
                'r_squared': result.r_squared,
                'num_trades': result.num_trades,
                'temporary_impact': result.temporary_impact,
                'permanent_impact': result.permanent_impact
            })
        
        return pd.DataFrame(results)


def compare_kyle_vs_amihud(
    trades_df: pd.DataFrame,
    quotes_df: pd.DataFrame
) -> Dict:
    """
    Compare Kyle's lambda with Amihud illiquidity to show why Kyle
    is more appropriate for nanosecond data.
    """
    from amihud_metrics import AmihudCalculator, AggregationLevel
    
    results = {}
    
    # Calculate Kyle's lambda (no aggregation needed!)
    print("Calculating Kyle's lambda (works directly on nanosecond data)...")
    kyle_result = KyleLambdaCalculator.calculate_kyle_lambda(
        trades_df, quotes_df
    )
    results['kyle_lambda'] = kyle_result.lambda_value
    results['kyle_r_squared'] = kyle_result.r_squared
    
    # Calculate intraday Kyle lambda profile
    print("Calculating intraday Kyle lambda profile...")
    intraday_kyle = KyleLambdaCalculator.calculate_intraday_kyle_lambda(
        trades_df, quotes_df
    )
    results['kyle_intraday'] = intraday_kyle
    
    # Calculate Amihud at different aggregations
    print("Calculating Amihud (requires aggregation)...")
    
    # Daily Amihud (original)
    daily_amihud = AmihudCalculator.calculate_amihud_original(trades_df)
    results['amihud_daily'] = daily_amihud
    
    # 5-minute Amihud
    amihud_5min = AmihudCalculator.calculate_amihud_intraday(
        trades_df, AggregationLevel.MINUTE_5
    )
    results['amihud_5min_mean'] = amihud_5min['amihud'].mean()
    results['amihud_5min_std'] = amihud_5min['amihud'].std()
    
    # Try tick-level Amihud (will fail or give nonsense)
    try:
        # This would be wrong!
        tick_returns = trades_df['price'].pct_change()
        tick_dollar_volume = trades_df['price'] * trades_df['volume']
        tick_amihud = (tick_returns.abs() / (tick_dollar_volume / 1e6)).mean()
        results['amihud_tick_wrong'] = tick_amihud
        print("WARNING: Tick-level Amihud is meaningless due to microstructure noise!")
    except:
        results['amihud_tick_wrong'] = np.nan
    
    return results


def visualize_kyle_lambda_analysis(
    trades_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    comparison_results: Dict
):
    """
    Visualize Kyle's lambda analysis and comparison with Amihud.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Kyle Lambda vs Amihud for Nanosecond Data', fontsize=16)
    
    # Plot 1: Intraday Kyle Lambda
    ax = axes[0, 0]
    if 'kyle_intraday' in comparison_results and len(comparison_results['kyle_intraday']) > 0:
        intraday = comparison_results['kyle_intraday']
        # Convert nanoseconds to hours for display
        intraday['hour'] = (intraday['timestamp_ns'] % (24 * 3600 * 1e9)) / (3600 * 1e9)
        ax.plot(intraday['hour'], intraday['lambda'], 'b-', alpha=0.7)
        ax.set_title('Kyle λ Throughout Trading Day')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Lambda (Price Impact)')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: R-squared of Kyle regression
    ax = axes[0, 1]
    if 'kyle_intraday' in comparison_results and len(comparison_results['kyle_intraday']) > 0:
        ax.plot(intraday['hour'], intraday['r_squared'], 'g-', alpha=0.7)
        ax.set_title('Kyle Regression Quality (R²)')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('R-squared')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Trade-by-trade impact
    ax = axes[0, 2]
    # Calculate immediate price impacts
    trades_with_direction = trades_df.copy()
    trades_with_direction['is_buy'] = KyleLambdaCalculator.determine_trade_direction(
        trades_df, quotes_df
    )
    trades_with_direction['signed_volume'] = np.where(
        trades_with_direction['is_buy'],
        trades_with_direction['volume'],
        -trades_with_direction['volume']
    )
    
    # Sample for visualization
    sample_size = min(1000, len(trades_with_direction))
    sample = trades_with_direction.sample(sample_size)
    
    ax.scatter(sample['signed_volume'], 
              sample['price'].diff(), 
              alpha=0.3, s=1)
    
    # Add regression line
    if comparison_results['kyle_lambda'] != 0:
        x_range = np.linspace(sample['signed_volume'].min(), 
                             sample['signed_volume'].max(), 100)
        y_pred = comparison_results['kyle_lambda'] * x_range
        ax.plot(x_range, y_pred, 'r-', label=f'λ = {comparison_results["kyle_lambda"]:.6f}')
    
    ax.set_title('Price Impact vs Signed Volume')
    ax.set_xlabel('Signed Volume')
    ax.set_ylabel('Price Change')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Comparison table
    ax = axes[1, 0]
    ax.axis('off')
    
    comparison_text = f"""
    KYLE vs AMIHUD COMPARISON
    {'='*30}
    
    Kyle Lambda:
    • Value: {comparison_results.get('kyle_lambda', 0):.6e}
    • R²: {comparison_results.get('kyle_r_squared', 0):.3f}
    • Works at: Nanosecond level
    • Aggregation: NOT REQUIRED
    
    Amihud Daily:
    • Value: {comparison_results.get('amihud_daily', 0):.6e}
    • Works at: Daily level
    • Aggregation: Full day
    
    Amihud 5-min:
    • Mean: {comparison_results.get('amihud_5min_mean', 0):.6e}
    • Std: {comparison_results.get('amihud_5min_std', 0):.6e}
    • Aggregation: 5 minutes
    
    Amihud Tick (WRONG):
    • Value: {comparison_results.get('amihud_tick_wrong', 0):.6e}
    • Issue: Microstructure noise
    """
    
    ax.text(0.1, 0.9, comparison_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Plot 5: Temporary vs Permanent Impact
    ax = axes[1, 1]
    if 'kyle_intraday' in comparison_results and len(comparison_results['kyle_intraday']) > 0:
        intraday = comparison_results['kyle_intraday']
        ax.plot(intraday['hour'], intraday['temporary_impact'], 
               'b-', label='Temporary', alpha=0.7)
        ax.plot(intraday['hour'], intraday['permanent_impact'], 
               'r-', label='Permanent', alpha=0.7)
        ax.set_title('Impact Decomposition')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Impact')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Why Kyle > Amihud for HFT
    ax = axes[1, 2]
    ax.axis('off')
    
    advantages_text = """
    WHY KYLE > AMIHUD FOR NANOSECOND DATA
    
    ✓ No aggregation required
    ✓ Captures immediate price impact
    ✓ Works at trade level
    ✓ Separates temporary/permanent
    ✓ Direct market depth measure
    ✓ Used in actual HFT models
    
    AMIHUD ISSUES WITH HFT:
    
    ✗ Requires aggregation (loses info)
    ✗ Microstructure noise dominates
    ✗ Bid-ask bounce contaminates
    ✗ Zero returns at tick level
    ✗ Designed for daily data
    """
    
    ax.text(0.1, 0.9, advantages_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('kyle_vs_amihud_comparison.png', dpi=150)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic nanosecond data for demonstration
    np.random.seed(42)
    n_trades = 10000
    n_quotes = 50000
    
    # Generate trades
    trade_times = np.sort(np.random.randint(0, 23400 * 1e9, n_trades))
    trade_prices = 100 * np.exp(np.cumsum(np.random.randn(n_trades) * 0.0001))
    trade_volumes = np.random.pareto(2, n_trades) * 100
    
    trades_df = pd.DataFrame({
        'timestamp_ns': trade_times,
        'price': trade_prices,
        'volume': trade_volumes
    })
    
    # Generate quotes (more frequent than trades)
    quote_times = np.sort(np.random.randint(0, 23400 * 1e9, n_quotes))
    spreads = np.random.exponential(0.01, n_quotes)
    mid_prices = np.interp(quote_times, trade_times, trade_prices)
    
    quotes_df = pd.DataFrame({
        'timestamp_ns': quote_times,
        'bid_price': mid_prices - spreads/2,
        'ask_price': mid_prices + spreads/2,
        'bid_size': np.random.randint(100, 1000, n_quotes),
        'ask_size': np.random.randint(100, 1000, n_quotes)
    })
    
    # Calculate and compare
    print("=" * 60)
    print("KYLE LAMBDA vs AMIHUD COMPARISON")
    print("=" * 60)
    
    comparison = compare_kyle_vs_amihud(trades_df, quotes_df)
    
    print(f"\nKyle Lambda: {comparison['kyle_lambda']:.6e}")
    print(f"Kyle R²: {comparison['kyle_r_squared']:.3f}")
    print(f"Amihud Daily: {comparison['amihud_daily']:.6e}")
    print(f"Amihud 5-min: {comparison['amihud_5min_mean']:.6e}")
    
    # Visualize
    visualize_kyle_lambda_analysis(trades_df, quotes_df, comparison)
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Kyle's lambda works directly on nanosecond data")
    print("while Amihud requires aggregation that loses information!")
    print("=" * 60)
