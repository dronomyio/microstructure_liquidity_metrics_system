import asyncio
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import redis
import asyncpg
from datetime import datetime, timedelta
import liquidity_metrics  # Our C++ bindings
from polygon_client import PolygonClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Liquidity Metrics API", version="1.0.0")

# Initialize connections
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
db_pool = None

class MetricsRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    metrics: List[str] = ["amihud", "duration", "hazard", "holes"]
    use_gpu: bool = True
    cache_results: bool = True

class MetricsResponse(BaseModel):
    symbol: str
    timestamp: datetime
    metrics: Dict[str, Any]
    processing_time_ms: float

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(
        host='postgres',
        database='liquidity_metrics',
        user='liquidity_user',
        password='changeme',
        min_size=10,
        max_size=20
    )
    
    # Initialize CUDA devices
    liquidity_metrics.initialize_cuda()
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()

@app.post("/calculate", response_model=MetricsResponse)
async def calculate_metrics(request: MetricsRequest, background_tasks: BackgroundTasks):
    """
    Calculate liquidity metrics for a given symbol and date range.
    """
    start_time = asyncio.get_event_loop().time()
    
    # Check cache first
    cache_key = f"metrics:{request.symbol}:{request.start_date}:{request.end_date}"
    if request.cache_results:
        cached = redis_client.get(cache_key)
        if cached:
            return JSONResponse(content=eval(cached))
    
    try:
        # Fetch data from Polygon
        polygon = PolygonClient()
        trades_data = await polygon.get_trades(
            request.symbol, 
            request.start_date, 
            request.end_date
        )
        quotes_data = await polygon.get_quotes(
            request.symbol,
            request.start_date,
            request.end_date
        )
        
        # Convert to numpy arrays for C++ processing
        trades_array = np.array(trades_data, dtype=[
            ('timestamp_ns', 'i8'),
            ('price', 'f8'),
            ('volume', 'f8'),
            ('conditions', 'i4'),
            ('exchange', 'S1')
        ])
        
        quotes_array = np.array(quotes_data, dtype=[
            ('timestamp_ns', 'i8'),
            ('bid_price', 'f8'),
            ('ask_price', 'f8'),
            ('bid_size', 'i4'),
            ('ask_size', 'i4'),
            ('bid_exchange', 'S1'),
            ('ask_exchange', 'S1')
        ])
        
        # Call C++ implementation
        if request.use_gpu:
            results = liquidity_metrics.calculate_gpu(
                trades_array,
                quotes_array,
                request.metrics
            )
        else:
            results = liquidity_metrics.calculate_cpu(
                trades_array,
                quotes_array,
                request.metrics
            )
        
        # Store in database
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO metrics_results 
                (symbol, timestamp, metrics_data, processing_time_ms)
                VALUES ($1, $2, $3, $4)
            """, request.symbol, datetime.now(), results, 
                (asyncio.get_event_loop().time() - start_time) * 1000)
        
        response = MetricsResponse(
            symbol=request.symbol,
            timestamp=datetime.now(),
            metrics=results,
            processing_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
        )
        
        # Cache results
        if request.cache_results:
            background_tasks.add_task(
                redis_client.setex,
                cache_key,
                3600,  # 1 hour TTL
                str(response.dict())
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/benchmark")
async def benchmark():
    """
    Run performance benchmark comparing CPU vs GPU.
    """
    # Generate synthetic data for benchmarking
    n_trades = 10_000_000
    synthetic_trades = np.random.random((n_trades, 5))
    
    # CPU benchmark
    cpu_start = asyncio.get_event_loop().time()
    cpu_results = liquidity_metrics.calculate_cpu(synthetic_trades, None, ["amihud"])
    cpu_time = asyncio.get_event_loop().time() - cpu_start
    
    # GPU benchmark
    gpu_start = asyncio.get_event_loop().time()
    gpu_results = liquidity_metrics.calculate_gpu(synthetic_trades, None, ["amihud"])
    gpu_time = asyncio.get_event_loop().time() - gpu_start
    
    return {
        "n_trades": n_trades,
        "cpu_time_ms": cpu_time * 1000,
        "gpu_time_ms": gpu_time * 1000,
        "speedup": cpu_time / gpu_time,
        "throughput_cpu": n_trades / cpu_time,
        "throughput_gpu": n_trades / gpu_time
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    cuda_available = liquidity_metrics.cuda_available()
    num_gpus = liquidity_metrics.get_num_gpus() if cuda_available else 0
    
    return {
        "status": "healthy",
        "cuda_available": cuda_available,
        "num_gpus": num_gpus,
        "simd_support": liquidity_metrics.get_simd_support()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
