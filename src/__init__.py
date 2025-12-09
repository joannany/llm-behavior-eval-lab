"""
LLM Behavior Evaluation Lab

A comprehensive framework for evaluating and monitoring Large Language Model 
behavior, focusing on safety alignment, capability assessment, and behavioral 
drift detection.
"""

__version__ = "0.1.0"
__author__ = "Anna"

from .evaluators import (
    BehaviorEvaluator,
    SafetyEvaluator,
    CapabilityEvaluator,
    ConsistencyEvaluator,
    BiasEvaluator,
)

from .monitors import (
    DriftMonitor,
    PerformanceMonitor,
    SafetyMonitor,
)

from .metrics import (
    SafetyMetrics,
    QualityMetrics,
    StatisticalMetrics,
    CapabilityMetrics,
)

from .utils import (
    ModelInterface,
    ReportGenerator,
    EvaluationPipeline,
)

__all__ = [
    # Version
    "__version__",
    
    # Evaluators
    "BehaviorEvaluator",
    "SafetyEvaluator", 
    "CapabilityEvaluator",
    "ConsistencyEvaluator",
    "BiasEvaluator",
    
    # Monitors
    "DriftMonitor",
    "PerformanceMonitor",
    "SafetyMonitor",
    
    # Metrics
    "SafetyMetrics",
    "QualityMetrics",
    "StatisticalMetrics",
    "CapabilityMetrics",
    
    # Utilities
    "ModelInterface",
    "ReportGenerator",
    "EvaluationPipeline",
]
