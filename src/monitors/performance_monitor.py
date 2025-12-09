"""
Performance Monitor

Tracks latency, token usage, and reliability metrics for LLM deployments.
"""

from dataclasses import dataclass, field
from typing import Any
import logging
from datetime import datetime
import time

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a monitoring window."""
    timestamp: datetime
    request_count: int
    success_count: int
    error_count: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_tokens_in: float
    mean_tokens_out: float
    throughput_rps: float
    error_rate: float


class PerformanceMonitor:
    """
    Monitors LLM performance metrics.
    
    Tracks latency, throughput, error rates, and token usage
    for production monitoring and alerting.
    """
    
    def __init__(
        self,
        window_seconds: int = 60,
        latency_threshold_ms: float = 1000.0,
        error_rate_threshold: float = 0.01,
        config: dict | None = None,
    ):
        self.window_seconds = window_seconds
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        self.config = config or {}
        
        self._latencies: list[float] = []
        self._tokens_in: list[int] = []
        self._tokens_out: list[int] = []
        self._errors: list[str] = []
        self._request_times: list[float] = []
        
        logger.info("Initialized PerformanceMonitor")
    
    def record_request(
        self,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record a single request."""
        self._latencies.append(latency_ms)
        self._tokens_in.append(tokens_in)
        self._tokens_out.append(tokens_out)
        self._request_times.append(time.time())
        
        if not success and error:
            self._errors.append(error)
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        current_time = time.time()
        cutoff = current_time - self.window_seconds
        
        # Filter to current window
        valid_indices = [
            i for i, t in enumerate(self._request_times) if t >= cutoff
        ]
        
        latencies = [self._latencies[i] for i in valid_indices]
        tokens_in = [self._tokens_in[i] for i in valid_indices]
        tokens_out = [self._tokens_out[i] for i in valid_indices]
        
        n = len(latencies)
        
        if n == 0:
            return PerformanceMetrics(
                timestamp=datetime.now(),
                request_count=0,
                success_count=0,
                error_count=0,
                mean_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                mean_tokens_in=0,
                mean_tokens_out=0,
                throughput_rps=0,
                error_rate=0,
            )
        
        sorted_latencies = np.sort(latencies)
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            request_count=n,
            success_count=n - len(self._errors),
            error_count=len(self._errors),
            mean_latency_ms=np.mean(latencies),
            p50_latency_ms=np.percentile(sorted_latencies, 50),
            p95_latency_ms=np.percentile(sorted_latencies, 95),
            p99_latency_ms=np.percentile(sorted_latencies, 99),
            mean_tokens_in=np.mean(tokens_in),
            mean_tokens_out=np.mean(tokens_out),
            throughput_rps=n / self.window_seconds,
            error_rate=len(self._errors) / n if n > 0 else 0,
        )
    
    def check_health(self) -> dict:
        """Check overall health status."""
        metrics = self.get_metrics()
        
        issues = []
        if metrics.mean_latency_ms > self.latency_threshold_ms:
            issues.append(f"High latency: {metrics.mean_latency_ms:.0f}ms")
        if metrics.error_rate > self.error_rate_threshold:
            issues.append(f"High error rate: {metrics.error_rate:.2%}")
        
        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "metrics": metrics,
        }
