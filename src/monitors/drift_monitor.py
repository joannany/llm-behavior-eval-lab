"""
Drift Monitor

Monitors and detects behavioral drift in LLM outputs over time,
using statistical tests and distribution comparison methods.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import logging
from datetime import datetime
import json

import numpy as np

logger = logging.getLogger(__name__)


class StatisticalTests(Enum):
    """Available statistical tests for drift detection."""
    KOLMOGOROV_SMIRNOV = "ks_test"
    WASSERSTEIN = "wasserstein"
    MMD = "mmd"  # Maximum Mean Discrepancy
    JS_DIVERGENCE = "js_divergence"  # Jensen-Shannon
    KL_DIVERGENCE = "kl_divergence"  # Kullback-Leibler
    CHI_SQUARE = "chi_square"
    ANDERSON_DARLING = "anderson_darling"
    POPULATION_STABILITY_INDEX = "psi"


@dataclass
class DriftResult:
    """Result of a drift detection analysis."""
    is_significant: bool
    overall_drift_score: float
    test_results: dict[str, float]
    timestamp: datetime
    samples_compared: int
    top_drifting_dimensions: list[int] = field(default_factory=list)
    confidence_level: float = 0.95
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "is_significant": self.is_significant,
            "overall_drift_score": self.overall_drift_score,
            "test_results": self.test_results,
            "timestamp": self.timestamp.isoformat(),
            "samples_compared": self.samples_compared,
            "top_drifting_dimensions": self.top_drifting_dimensions,
            "confidence_level": self.confidence_level,
            "details": self.details,
        }


@dataclass
class DriftAlert:
    """Alert generated when significant drift is detected."""
    alert_id: str
    severity: str  # "low", "medium", "high", "critical"
    drift_result: DriftResult
    message: str
    recommended_actions: list[str]
    created_at: datetime


class StatisticalTestRunner:
    """Runs statistical tests for drift detection."""
    
    def __init__(self):
        self._test_functions: dict[StatisticalTests, Callable] = {
            StatisticalTests.KOLMOGOROV_SMIRNOV: self._ks_test,
            StatisticalTests.WASSERSTEIN: self._wasserstein_distance,
            StatisticalTests.JS_DIVERGENCE: self._js_divergence,
            StatisticalTests.POPULATION_STABILITY_INDEX: self._psi,
        }
    
    def run_test(
        self,
        test: StatisticalTests,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> tuple[float, float]:
        """
        Run a statistical test.
        
        Args:
            test: The statistical test to run
            baseline: Baseline distribution samples
            current: Current distribution samples
            
        Returns:
            Tuple of (statistic, p_value)
        """
        if test not in self._test_functions:
            raise NotImplementedError(f"Test {test} not implemented")
        
        return self._test_functions[test](baseline, current)
    
    def _ks_test(
        self, 
        baseline: np.ndarray, 
        current: np.ndarray
    ) -> tuple[float, float]:
        """Kolmogorov-Smirnov test."""
        # Simple implementation without scipy
        n1, n2 = len(baseline), len(current)
        
        # Sort both arrays
        sorted_baseline = np.sort(baseline.flatten())
        sorted_current = np.sort(current.flatten())
        
        # Combine and sort all values
        all_values = np.sort(np.concatenate([sorted_baseline, sorted_current]))
        
        # Calculate CDFs at each point
        cdf1 = np.searchsorted(sorted_baseline, all_values, side='right') / n1
        cdf2 = np.searchsorted(sorted_current, all_values, side='right') / n2
        
        # KS statistic is the maximum difference
        ks_stat = np.max(np.abs(cdf1 - cdf2))
        
        # Approximate p-value (simplified)
        n = n1 * n2 / (n1 + n2)
        p_value = np.exp(-2 * n * ks_stat ** 2)
        
        return ks_stat, p_value
    
    def _wasserstein_distance(
        self, 
        baseline: np.ndarray, 
        current: np.ndarray
    ) -> tuple[float, float]:
        """Wasserstein (Earth Mover's) distance."""
        sorted_baseline = np.sort(baseline.flatten())
        sorted_current = np.sort(current.flatten())
        
        # Interpolate to same length
        n = max(len(sorted_baseline), len(sorted_current))
        baseline_interp = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(sorted_baseline)),
            sorted_baseline
        )
        current_interp = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(sorted_current)),
            sorted_current
        )
        
        # Wasserstein distance is mean absolute difference of sorted values
        distance = np.mean(np.abs(baseline_interp - current_interp))
        
        # No p-value for Wasserstein, return distance as proxy
        return distance, 1.0 - min(distance, 1.0)
    
    def _js_divergence(
        self, 
        baseline: np.ndarray, 
        current: np.ndarray
    ) -> tuple[float, float]:
        """Jensen-Shannon divergence."""
        # Create histograms
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bins = np.linspace(min_val, max_val, 50)
        
        p, _ = np.histogram(baseline, bins=bins, density=True)
        q, _ = np.histogram(current, bins=bins, density=True)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p = p + eps
        q = q + eps
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Calculate JS divergence
        m = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        
        return js, 1.0 - min(js, 1.0)
    
    def _psi(
        self, 
        baseline: np.ndarray, 
        current: np.ndarray
    ) -> tuple[float, float]:
        """Population Stability Index."""
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bins = np.linspace(min_val, max_val, 10)
        
        baseline_counts, _ = np.histogram(baseline, bins=bins)
        current_counts, _ = np.histogram(current, bins=bins)
        
        # Convert to percentages
        baseline_pct = baseline_counts / baseline_counts.sum()
        current_pct = current_counts / current_counts.sum()
        
        # Add small epsilon
        eps = 1e-10
        baseline_pct = baseline_pct + eps
        current_pct = current_pct + eps
        
        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        
        # PSI interpretation: < 0.1 no change, 0.1-0.25 slight change, > 0.25 significant
        if psi < 0.1:
            p_value = 0.9
        elif psi < 0.25:
            p_value = 0.5
        else:
            p_value = 0.01
        
        return psi, p_value


class DriftDetector:
    """
    Detects drift between baseline and current distributions.
    
    Uses multiple statistical tests to identify significant
    changes in model behavior.
    
    Example:
        >>> detector = DriftDetector(
        ...     baseline_embeddings=baseline_data,
        ...     statistical_tests=[StatisticalTests.KS_TEST],
        ...     significance_level=0.05
        ... )
        >>> result = detector.check(new_data)
    """
    
    def __init__(
        self,
        baseline_embeddings: np.ndarray | None = None,
        statistical_tests: list[StatisticalTests] | None = None,
        significance_level: float = 0.05,
        config: dict | None = None,
    ):
        """
        Initialize drift detector.
        
        Args:
            baseline_embeddings: Baseline distribution data
            statistical_tests: Tests to run
            significance_level: P-value threshold for significance
            config: Additional configuration
        """
        self.baseline = baseline_embeddings
        self.tests = statistical_tests or [
            StatisticalTests.KOLMOGOROV_SMIRNOV,
            StatisticalTests.WASSERSTEIN,
        ]
        self.significance_level = significance_level
        self.config = config or {}
        self.test_runner = StatisticalTestRunner()
        
        logger.info(
            f"Initialized DriftDetector with {len(self.tests)} tests"
        )
    
    def set_baseline(self, baseline: np.ndarray) -> None:
        """Set or update the baseline distribution."""
        self.baseline = baseline
        logger.info(f"Baseline updated with shape {baseline.shape}")
    
    def check(self, current: np.ndarray) -> DriftResult:
        """
        Check for drift between baseline and current data.
        
        Args:
            current: Current distribution data
            
        Returns:
            DriftResult with comprehensive drift analysis
        """
        if self.baseline is None:
            raise ValueError("Baseline not set. Call set_baseline first.")
        
        test_results = {}
        significant_tests = 0
        
        # Run each statistical test
        for test in self.tests:
            try:
                statistic, p_value = self.test_runner.run_test(
                    test, self.baseline, current
                )
                test_results[test.value] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": p_value < self.significance_level,
                }
                if p_value < self.significance_level:
                    significant_tests += 1
            except Exception as e:
                logger.error(f"Error running {test.value}: {e}")
                test_results[test.value] = {"error": str(e)}
        
        # Calculate overall drift score
        drift_scores = [
            r["statistic"] for r in test_results.values() 
            if "statistic" in r
        ]
        overall_drift = np.mean(drift_scores) if drift_scores else 0.0
        
        # Identify drifting dimensions (for multi-dimensional data)
        top_dimensions = self._find_drifting_dimensions(current)
        
        # Determine if drift is significant
        is_significant = significant_tests >= len(self.tests) / 2
        
        return DriftResult(
            is_significant=is_significant,
            overall_drift_score=overall_drift,
            test_results={k: v.get("statistic", 0) for k, v in test_results.items()},
            timestamp=datetime.now(),
            samples_compared=len(current),
            top_drifting_dimensions=top_dimensions,
            confidence_level=1 - self.significance_level,
            details={
                "full_test_results": test_results,
                "significant_tests": significant_tests,
                "total_tests": len(self.tests),
            },
        )
    
    def _find_drifting_dimensions(
        self, 
        current: np.ndarray, 
        top_k: int = 5
    ) -> list[int]:
        """Find dimensions with most drift."""
        if self.baseline is None or len(self.baseline.shape) == 1:
            return []
        
        if len(self.baseline.shape) > 1 and len(current.shape) > 1:
            # Calculate per-dimension drift
            n_dims = min(self.baseline.shape[-1], current.shape[-1])
            dimension_drifts = []
            
            for dim in range(n_dims):
                baseline_dim = self.baseline[..., dim].flatten()
                current_dim = current[..., dim].flatten()
                
                stat, _ = self.test_runner.run_test(
                    StatisticalTests.KOLMOGOROV_SMIRNOV,
                    baseline_dim,
                    current_dim,
                )
                dimension_drifts.append((dim, stat))
            
            # Sort by drift score and return top k
            sorted_dims = sorted(dimension_drifts, key=lambda x: x[1], reverse=True)
            return [dim for dim, _ in sorted_dims[:top_k]]
        
        return []


class DriftMonitor:
    """
    Continuous monitoring system for behavioral drift.
    
    Tracks model behavior over time and generates alerts
    when significant drift is detected.
    
    Example:
        >>> monitor = DriftMonitor(
        ...     baseline_distribution=baseline_responses,
        ...     drift_threshold=0.05,
        ... )
        >>> drift_report = monitor.detect_drift(current_responses)
    """
    
    def __init__(
        self,
        baseline_distribution: np.ndarray | None = None,
        drift_threshold: float = 0.05,
        statistical_test: StatisticalTests = StatisticalTests.KOLMOGOROV_SMIRNOV,
        window_size: int = 1000,
        alert_callback: Callable[[DriftAlert], None] | None = None,
        config: dict | None = None,
    ):
        """
        Initialize drift monitor.
        
        Args:
            baseline_distribution: Baseline data
            drift_threshold: Threshold for drift significance
            statistical_test: Primary statistical test to use
            window_size: Rolling window size for monitoring
            alert_callback: Function to call when drift is detected
            config: Additional configuration
        """
        self.detector = DriftDetector(
            baseline_embeddings=baseline_distribution,
            statistical_tests=[statistical_test],
            significance_level=drift_threshold,
        )
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.alert_callback = alert_callback
        self.config = config or {}
        
        # State tracking
        self._history: list[DriftResult] = []
        self._rolling_window: list[np.ndarray] = []
        self._alerts: list[DriftAlert] = []
        
        logger.info(
            f"Initialized DriftMonitor with threshold={drift_threshold}"
        )
    
    def detect_drift(
        self, 
        current_data: np.ndarray
    ) -> DriftResult:
        """
        Detect drift in current data.
        
        Args:
            current_data: Current distribution data
            
        Returns:
            DriftResult with analysis
        """
        result = self.detector.check(current_data)
        self._history.append(result)
        
        # Update rolling window
        self._rolling_window.append(current_data)
        if len(self._rolling_window) > 10:  # Keep last 10 windows
            self._rolling_window.pop(0)
        
        # Generate alert if drift is significant
        if result.is_significant:
            alert = self._generate_alert(result)
            self._alerts.append(alert)
            
            if self.alert_callback:
                self.alert_callback(alert)
        
        return result
    
    def _generate_alert(self, result: DriftResult) -> DriftAlert:
        """Generate an alert from a drift result."""
        import uuid
        
        # Determine severity
        if result.overall_drift_score > 0.5:
            severity = "critical"
        elif result.overall_drift_score > 0.3:
            severity = "high"
        elif result.overall_drift_score > 0.15:
            severity = "medium"
        else:
            severity = "low"
        
        # Generate recommendations
        recommendations = [
            "Review recent model changes or deployments",
            "Check for data distribution shifts in inputs",
            "Consider retraining or recalibrating the model",
        ]
        
        if severity in ["critical", "high"]:
            recommendations.insert(0, "URGENT: Consider rolling back recent changes")
        
        return DriftAlert(
            alert_id=str(uuid.uuid4())[:8],
            severity=severity,
            drift_result=result,
            message=f"Significant drift detected (score={result.overall_drift_score:.3f})",
            recommended_actions=recommendations,
            created_at=datetime.now(),
        )
    
    def get_history(self) -> list[DriftResult]:
        """Get drift detection history."""
        return self._history.copy()
    
    def get_alerts(self) -> list[DriftAlert]:
        """Get all generated alerts."""
        return self._alerts.copy()
    
    def get_trend(self, window: int = 10) -> dict:
        """
        Get drift trend over recent history.
        
        Args:
            window: Number of recent results to analyze
            
        Returns:
            Trend analysis dictionary
        """
        recent = self._history[-window:] if len(self._history) >= window else self._history
        
        if not recent:
            return {"error": "No history available"}
        
        scores = [r.overall_drift_score for r in recent]
        
        return {
            "window_size": len(recent),
            "mean_drift": np.mean(scores),
            "max_drift": np.max(scores),
            "min_drift": np.min(scores),
            "trend_direction": "increasing" if len(scores) > 1 and scores[-1] > scores[0] else "stable",
            "alerts_in_window": sum(1 for r in recent if r.is_significant),
        }
    
    def export_report(self, output_path: str) -> None:
        """Export monitoring report to file."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_checks": len(self._history),
            "total_alerts": len(self._alerts),
            "trend": self.get_trend(),
            "recent_results": [r.to_dict() for r in self._history[-20:]],
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report exported to {output_path}")
