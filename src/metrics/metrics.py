"""
Metrics Module

Comprehensive metrics for LLM evaluation including safety, quality,
statistical, and capability metrics.
"""

from typing import Any
import numpy as np

from ..evaluators.behavior_evaluator import MetricCalculator, EvaluationResult


class SafetyMetrics(MetricCalculator):
    """
    Calculates safety-related metrics from evaluation results.
    
    Metrics include:
    - harm_score: Probability of harmful content generation
    - refusal_rate: Rate of appropriate refusals
    - jailbreak_resistance: Resistance to adversarial prompts
    - boundary_adherence: Compliance with safety guidelines
    """
    
    @property
    def name(self) -> str:
        return "safety_metrics"
    
    def calculate(self, results: list[EvaluationResult]) -> dict:
        """Calculate safety metrics from results."""
        if not results:
            return {}
        
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        
        refusals = 0
        harm_detected = 0
        
        for r in results:
            if hasattr(r, "refusal_detected") and r.refusal_detected:
                refusals += 1
            if hasattr(r, "harmful_content_detected") and r.harmful_content_detected:
                harm_detected += 1
        
        return {
            "pass_rate": passed / total,
            "refusal_rate": refusals / total,
            "harm_score": harm_detected / total,
            "boundary_adherence": passed / total,
            "mean_safety_score": np.mean([r.score for r in results]),
        }


class QualityMetrics(MetricCalculator):
    """
    Calculates quality metrics including coherence, relevance, and factuality.
    """
    
    @property
    def name(self) -> str:
        return "quality_metrics"
    
    def calculate(self, results: list[EvaluationResult]) -> dict:
        """Calculate quality metrics from results."""
        if not results:
            return {}
        
        scores = [r.score for r in results]
        
        correctness = []
        reasoning = []
        completeness = []
        
        for r in results:
            if hasattr(r, "correctness_score"):
                correctness.append(r.correctness_score)
            if hasattr(r, "reasoning_score"):
                reasoning.append(r.reasoning_score)
            if hasattr(r, "completeness_score"):
                completeness.append(r.completeness_score)
        
        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "mean_correctness": np.mean(correctness) if correctness else None,
            "mean_reasoning": np.mean(reasoning) if reasoning else None,
            "mean_completeness": np.mean(completeness) if completeness else None,
        }


class StatisticalMetrics(MetricCalculator):
    """
    Calculates statistical metrics for distribution analysis.
    """
    
    @property
    def name(self) -> str:
        return "statistical_metrics"
    
    def calculate(self, results: list[EvaluationResult]) -> dict:
        """Calculate statistical metrics from results."""
        if not results:
            return {}
        
        scores = np.array([r.score for r in results])
        latencies = np.array([r.latency_ms for r in results])
        
        return {
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_median": float(np.median(scores)),
            "score_skewness": float(self._skewness(scores)),
            "score_kurtosis": float(self._kurtosis(scores)),
            "latency_mean": float(np.mean(latencies)),
            "latency_p50": float(np.percentile(latencies, 50)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "latency_p99": float(np.percentile(latencies, 99)),
        }
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)


class CapabilityMetrics(MetricCalculator):
    """
    Calculates capability metrics across reasoning and knowledge dimensions.
    """
    
    @property
    def name(self) -> str:
        return "capability_metrics"
    
    def calculate(self, results: list[EvaluationResult]) -> dict:
        """Calculate capability metrics from results."""
        if not results:
            return {}
        
        by_dimension: dict[str, list[float]] = {}
        by_difficulty: dict[str, list[float]] = {}
        
        for r in results:
            if hasattr(r, "dimension") and r.dimension:
                dim = r.dimension.value if hasattr(r.dimension, "value") else str(r.dimension)
                if dim not in by_dimension:
                    by_dimension[dim] = []
                by_dimension[dim].append(r.score)
            
            if hasattr(r, "difficulty") and r.difficulty:
                diff = r.difficulty.value if hasattr(r.difficulty, "value") else str(r.difficulty)
                if diff not in by_difficulty:
                    by_difficulty[diff] = []
                by_difficulty[diff].append(r.score)
        
        metrics = {
            "overall_score": np.mean([r.score for r in results]),
            "pass_rate": sum(1 for r in results if r.passed) / len(results),
        }
        
        for dim, scores in by_dimension.items():
            metrics[f"{dim}_score"] = np.mean(scores)
        
        for diff, scores in by_difficulty.items():
            metrics[f"{diff}_score"] = np.mean(scores)
        
        return metrics
