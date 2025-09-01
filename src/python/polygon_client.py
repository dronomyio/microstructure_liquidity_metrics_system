import aiohttp
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import logging
from urllib.parse import urljoin
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class PolygonClient:
    """
    Async client for Polygon.io flat files API.
    """
    
    BASE_URL = "https://files.polygon.io"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Polygon API key required")
        
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_trades(self, symbol: str, date: str) -> pd.DataFrame:
        """
        Fetch trades data for a symbol on a specific date.
        
        Args:
            symbol: Stock symbol
            date: Date in YYYY-MM-DD format
        
        Returns:
            DataFrame with nanosecond-precision trade data
        """
        url = f"{self.BASE_URL}/us/stocks/trades/{date}/{symbol}.parquet"
        
        async with self.session.get(
            url,
            params={"apikey": self.api_key}
        ) as response:
            if response.status == 200:
                content = await response.read()
                
                # Parse Parquet data
                df = pd.read_parquet(pd.io.common.BytesIO(content))
                
                # Convert SIP timestamp to nanoseconds
                df['timestamp_ns'] = df['sip_timestamp']
                
                # Extract required fields
                trades_df = pd.DataFrame({
                    'timestamp_ns': df['timestamp_ns'],
                    'price': df['price'],
                    'volume': df['size'],
                    'conditions': df['conditions'].apply(self._encode_conditions),
                    'exchange': df['exchange'].apply(ord)
                })
                
                return trades_df
            else:
                raise Exception(f"Failed to fetch trades: {response.status}")
    
    async def get_quotes(self, symbol: str, date: str) -> pd.DataFrame:
        """
        Fetch quotes data for a symbol on a specific date.
        
        Args:
            symbol: Stock symbol
            date: Date in YYYY-MM-DD format
        
        Returns:
            DataFrame with nanosecond-precision quote data
        """
        url = f"{self.BASE_URL}/us/stocks/quotes/{date}/{symbol}.parquet"
        
        async with self.session.get(
            url,
            params={"apikey": self.api_key}
        ) as response:
            if response.status == 200:
                content = await response.read()
                
                # Parse Parquet data
                df = pd.read_parquet(pd.io.common.BytesIO(content))
                
                # Convert SIP timestamp to nanoseconds
                df['timestamp_ns'] = df['sip_timestamp']
                
                # Extract required fields
                quotes_df = pd.DataFrame({
                    'timestamp_ns': df['timestamp_ns'],
                    'bid_price': df['bid_price'],
                    'ask_price': df['ask_price'],
                    'bid_size': df['bid_size'],
                    'ask_size': df['ask_size'],
                    'bid_exchange': df['bid_exchange'].apply(ord),
                    'ask_exchange': df['ask_exchange'].apply(ord)
                })
                
                return quotes_df
            else:
                raise Exception(f"Failed to fetch quotes: {response.status}")
    
    async def get_multiple_days(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        data_type: str = "trades"
    ) -> pd.DataFrame:
        """
        Fetch data for multiple days concurrently.
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='B')  # Business days only
        
        fetch_func = self.get_trades if data_type == "trades" else self.get_quotes
        
        tasks = [
            fetch_func(symbol, date.strftime('%Y-%m-%d'))
            for date in dates
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and concatenate
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        if valid_results:
            return pd.concat(valid_results, ignore_index=True)
        else:
            raise Exception("No valid data retrieved")
    
    @staticmethod
    def _encode_conditions(conditions_list):
        """Encode trade conditions as integer bitmap."""
        if not conditions_list:
            return 0
        
        # Map common conditions to bits
        condition_map = {
            'regular': 1 << 0,
            'cash': 1 << 1,
            'next_day': 1 << 2,
            'seller': 1 << 3,
            'bunched': 1 << 4,
            'spread': 1 << 5,
            'opening': 1 << 6,
            'closing': 1 << 7,
            'cross': 1 << 8,
            'derivatively_priced': 1 << 9,
            'prior_reference_price': 1 << 10,
            'extended_hours': 1 << 11,
            'odd_lot': 1 << 12
        }
        
        encoded = 0
        for condition in conditions_list:
            encoded |= condition_map.get(condition.lower(), 0)
        
        return encoded


# Example usage
async def main():
    async with PolygonClient() as client:
        # Fetch single day
        trades = await client.get_trades("AAPL", "2024-01-15")
        print(f"Retrieved {len(trades)} trades")
        
        # Fetch multiple days
        multi_day_trades = await client.get_multiple_days(
            "AAPL", 
            "2024-01-01", 
            "2024-01-31",
            "trades"
        )
        print(f"Retrieved {len(multi_day_trades)} trades for January 2024")

if __name__ == "__main__":
    asyncio.run(main())
