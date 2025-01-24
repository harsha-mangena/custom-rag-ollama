from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
import json
import logging
from config.settings import Settings
import aiosqlite

class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self, db_path: str = Settings.DB_PATH):
        """Initialize metrics collector."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

    async def store_query_metrics(
        self,
        query_text: str,
        embedding_model: str,
        n_results: int,
        similarity_threshold: float,
        metrics: Dict[str, float],
        result_count: int
    ):
        """Store query execution metrics."""
        try:
            data = {
                "query_text": query_text,
                "embedding_model": embedding_model,
                "n_results": n_results,
                "similarity_threshold": similarity_threshold,
                "result_count": result_count,
                **metrics
            }
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO metrics (type, data)
                    VALUES (?, ?)
                    """,
                    ("query", json.dumps(data))
                )
                await db.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store query metrics: {e}")

    async def store_benchmark_result(
        self,
        model_name: str,
        metrics: Dict[str, Any]
    ):
        """Store benchmark results."""
        try:
            data = {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                **metrics
            }
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO metrics (type, data)
                    VALUES (?, ?)
                    """,
                    ("benchmark", json.dumps(data))
                )
                await db.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store benchmark result: {e}")

    async def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of system performance."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get query metrics
                query_stats = []
                async with db.execute(
                    "SELECT data FROM metrics WHERE type = 'query'"
                ) as cursor:
                    async for row in cursor:
                        query_stats.append(json.loads(row[0]))
                
                # Get benchmark metrics
                benchmark_stats = []
                async with db.execute(
                    "SELECT data FROM metrics WHERE type = 'benchmark'"
                ) as cursor:
                    async for row in cursor:
                        benchmark_stats.append(json.loads(row[0]))
                
                # Calculate statistics
                query_df = pd.DataFrame(query_stats)
                benchmark_df = pd.DataFrame(benchmark_stats)
                
                stats = {
                    "total_queries": len(query_df),
                    "avg_query_time": query_df["total_time"].mean() if not query_df.empty else 0,
                    "avg_result_count": query_df["result_count"].mean() if not query_df.empty else 0,
                    "total_benchmarks": len(benchmark_df),
                    "model_performance": benchmark_df.groupby("model_name")["total_time"].mean().to_dict() if not benchmark_df.empty else {}
                }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get summary statistics: {e}")
            return {}