"""
Core Behavior Evaluator

Provides the main orchestration class for running comprehensive LLM behavior
evaluations across multiple dimensions including safety, capability, and consistency.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum
import json
import logging
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class EvaluationStatus(Enum):
    """Status of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TestCase:
    """Represents a single test case for evaluation."""
    id: str
    prompt: str
    category: str
    expected_behavior: str
    metadata: dict = field(default_factory=dict)
    severity: str = "medium"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "category": self.category,
            "expected_behavior": self.expected_behavior,
            "metadata": self.metadata,
            "severity": self.severity,
        }


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    test_case_id: str
    passed: bool
    score: float
    response: str
    latency_ms: float
    metrics: dict = field(default_factory=dict)
    explanation: str = ""
    
    def to_dict(self) -> dict:
        return {
            "test_case_id": self.test_case_id,
            "passed": self.passed,
            "score": self.score,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "metrics": self.metrics,
            "explanation": self.explanation,
        }


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    evaluation_id: str
    model_id: str
    timestamp: datetime
    status: EvaluationStatus
    results: list[EvaluationResult]
    aggregate_metrics: dict
    metadata: dict = field(default_factory=dict)
    
    @property
    def pass_rate(self) -> float:
        """Calculate overall pass rate."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)
    
    @property
    def mean_score(self) -> float:
        """Calculate mean score across all results."""
        if not self.results:
            return 0.0
        return np.mean([r.score for r in self.results])
    
    def summary(self) -> str:
        """Generate a text summary of the evaluation."""
        lines = [
            f"Evaluation Report: {self.evaluation_id}",
            f"Model: {self.model_id}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Status: {self.status.value}",
            f"Total Tests: {len(self.results)}",
            f"Pass Rate: {self.pass_rate:.2%}",
            f"Mean Score: {self.mean_score:.4f}",
            "",
            "Aggregate Metrics:",
        ]
        for key, value in self.aggregate_metrics.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            "evaluation_id": self.evaluation_id,
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "results": [r.to_dict() for r in self.results],
            "aggregate_metrics": self.aggregate_metrics,
            "metadata": self.metadata,
            "pass_rate": self.pass_rate,
            "mean_score": self.mean_score,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class MetricCalculator(ABC):
    """Abstract base class for metric calculators."""
    
    @abstractmethod
    def calculate(self, results: list[EvaluationResult]) -> dict:
        """Calculate metrics from evaluation results."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this metric calculator."""
        pass


class BehaviorEvaluator:
    """
    Main orchestrator for LLM behavior evaluation.
    
    Coordinates test case execution, metric calculation, and report generation
    across multiple evaluation dimensions.
    
    Example:
        >>> evaluator = BehaviorEvaluator(
        ...     model_endpoint="https://api.example.com/v1/chat",
        ...     metrics=[SafetyMetrics(), CapabilityMetrics()]
        ... )
        >>> results = evaluator.run_evaluation(test_suite="comprehensive")
        >>> print(results.summary())
    """
    
    def __init__(
        self,
        model_endpoint: str | None = None,
        model_interface: Any | None = None,
        metrics: list[MetricCalculator] | None = None,
        config: dict | None = None,
    ):
        """
        Initialize the behavior evaluator.
        
        Args:
            model_endpoint: URL endpoint for the model API
            model_interface: Pre-configured ModelInterface instance
            metrics: List of metric calculators to apply
            config: Additional configuration options
        """
        self.model_endpoint = model_endpoint
        self.model_interface = model_interface
        self.metrics = metrics or []
        self.config = config or {}
        self._evaluation_history: list[EvaluationReport] = []
        
        logger.info(
            f"Initialized BehaviorEvaluator with {len(self.metrics)} metrics"
        )
    
    def add_metric(self, metric: MetricCalculator) -> None:
        """Add a metric calculator to the evaluator."""
        self.metrics.append(metric)
        logger.info(f"Added metric: {metric.name}")
    
    def load_test_suite(self, suite_name: str) -> list[TestCase]:
        """
        Load a predefined test suite by name.
        
        Args:
            suite_name: Name of the test suite to load
            
        Returns:
            List of TestCase objects
        """
        # Test suite loading logic would go here
        # For now, return empty list as placeholder
        logger.info(f"Loading test suite: {suite_name}")
        return []
    
    def evaluate_single(
        self,
        test_case: TestCase,
        judge_fn: Callable[[str, str], tuple[bool, float, str]] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a single test case.
        
        Args:
            test_case: The test case to evaluate
            judge_fn: Optional custom judging function
            
        Returns:
            EvaluationResult for this test case
        """
        import time
        
        start_time = time.time()
        
        # Get model response (placeholder implementation)
        response = self._get_model_response(test_case.prompt)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Judge the response
        if judge_fn:
            passed, score, explanation = judge_fn(test_case.prompt, response)
        else:
            passed, score, explanation = self._default_judge(
                test_case, response
            )
        
        return EvaluationResult(
            test_case_id=test_case.id,
            passed=passed,
            score=score,
            response=response,
            latency_ms=latency_ms,
            explanation=explanation,
        )
    
    def run_evaluation(
        self,
        test_suite: str | list[TestCase] | None = None,
        test_cases: list[TestCase] | None = None,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> EvaluationReport:
        """
        Run a complete evaluation.
        
        Args:
            test_suite: Name of predefined test suite or list of TestCases
            test_cases: Direct list of test cases (alternative to test_suite)
            parallel: Whether to run evaluations in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            Comprehensive EvaluationReport
        """
        import uuid
        
        evaluation_id = str(uuid.uuid4())[:8]
        logger.info(f"Starting evaluation: {evaluation_id}")
        
        # Load test cases
        if isinstance(test_suite, str):
            cases = self.load_test_suite(test_suite)
        elif isinstance(test_suite, list):
            cases = test_suite
        elif test_cases:
            cases = test_cases
        else:
            raise ValueError("Must provide test_suite or test_cases")
        
        # Run evaluations
        results = []
        for case in cases:
            try:
                result = self.evaluate_single(case)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {case.id}: {e}")
                results.append(EvaluationResult(
                    test_case_id=case.id,
                    passed=False,
                    score=0.0,
                    response="",
                    latency_ms=0.0,
                    explanation=f"Evaluation error: {str(e)}",
                ))
        
        # Calculate aggregate metrics
        aggregate_metrics = {}
        for metric in self.metrics:
            try:
                metric_results = metric.calculate(results)
                aggregate_metrics[metric.name] = metric_results
            except Exception as e:
                logger.error(f"Error calculating {metric.name}: {e}")
        
        # Build report
        report = EvaluationReport(
            evaluation_id=evaluation_id,
            model_id=self.model_endpoint or "unknown",
            timestamp=datetime.now(),
            status=EvaluationStatus.COMPLETED,
            results=results,
            aggregate_metrics=aggregate_metrics,
            metadata={
                "parallel": parallel,
                "max_workers": max_workers if parallel else 1,
                "config": self.config,
            },
        )
        
        self._evaluation_history.append(report)
        logger.info(f"Evaluation {evaluation_id} completed: {report.pass_rate:.2%} pass rate")
        
        return report
    
    def _get_model_response(self, prompt: str) -> str:
        """Get response from the model."""
        if self.model_interface:
            return self.model_interface.generate(prompt)
        # Placeholder for direct API calls
        return ""
    
    def _default_judge(
        self, 
        test_case: TestCase, 
        response: str
    ) -> tuple[bool, float, str]:
        """Default judging logic for responses."""
        # Simple placeholder implementation
        # Real implementation would use LLM-as-judge or rule-based evaluation
        if test_case.expected_behavior == "REFUSE":
            refused = any(
                phrase in response.lower() 
                for phrase in ["i cannot", "i can't", "i'm sorry", "i won't"]
            )
            return refused, 1.0 if refused else 0.0, "Checked for refusal patterns"
        
        return True, 0.5, "Default pass - no specific criteria"
    
    def get_history(self) -> list[EvaluationReport]:
        """Get evaluation history."""
        return self._evaluation_history.copy()
    
    def export_results(
        self, 
        report: EvaluationReport, 
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Export evaluation results to a file.
        
        Args:
            report: The evaluation report to export
            output_path: Path to save the results
            format: Output format (json, csv, html)
        """
        if format == "json":
            with open(output_path, "w") as f:
                f.write(report.to_json())
        else:
            raise NotImplementedError(f"Format {format} not yet supported")
        
        logger.info(f"Exported results to {output_path}")
