"""
Evaluation Pipeline

Composable pipeline for building custom evaluation workflows
with multiple stages and evaluators.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """A single stage in the evaluation pipeline."""
    name: str
    evaluator: Any
    config: dict = field(default_factory=dict)
    enabled: bool = True


@dataclass
class PipelineResult:
    """Results from a pipeline run."""
    pipeline_id: str
    timestamp: datetime
    stages_run: int
    stage_results: dict[str, Any]
    overall_status: str
    duration_seconds: float


class EvaluationPipeline:
    """
    Composable evaluation pipeline.
    
    Allows building custom evaluation workflows by chaining
    multiple evaluators and processors.
    
    Example:
        >>> pipeline = EvaluationPipeline([
        ...     SafetyEvaluator(categories=["harmful", "bias"]),
        ...     CapabilityEvaluator(dimensions=["reasoning"]),
        ...     DriftMonitor(baseline="v1.0_baseline.json")
        ... ])
        >>> results = pipeline.evaluate(model, test_cases)
    """
    
    def __init__(
        self,
        stages: list[Any] | None = None,
        config: dict | None = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            stages: List of evaluator instances
            config: Pipeline configuration
        """
        self._stages: list[PipelineStage] = []
        self.config = config or {}
        
        if stages:
            for i, stage in enumerate(stages):
                self.add_stage(
                    name=f"stage_{i}_{type(stage).__name__}",
                    evaluator=stage,
                )
        
        logger.info(f"Initialized EvaluationPipeline with {len(self._stages)} stages")
    
    def add_stage(
        self,
        name: str,
        evaluator: Any,
        config: dict | None = None,
    ) -> "EvaluationPipeline":
        """
        Add a stage to the pipeline.
        
        Args:
            name: Stage name
            evaluator: Evaluator instance
            config: Stage-specific configuration
            
        Returns:
            Self for method chaining
        """
        stage = PipelineStage(
            name=name,
            evaluator=evaluator,
            config=config or {},
        )
        self._stages.append(stage)
        logger.info(f"Added stage: {name}")
        return self
    
    def remove_stage(self, name: str) -> "EvaluationPipeline":
        """Remove a stage by name."""
        self._stages = [s for s in self._stages if s.name != name]
        return self
    
    def enable_stage(self, name: str) -> "EvaluationPipeline":
        """Enable a stage by name."""
        for stage in self._stages:
            if stage.name == name:
                stage.enabled = True
        return self
    
    def disable_stage(self, name: str) -> "EvaluationPipeline":
        """Disable a stage by name."""
        for stage in self._stages:
            if stage.name == name:
                stage.enabled = False
        return self
    
    def evaluate(
        self,
        model_endpoint: str | None = None,
        model_interface: Any = None,
        test_cases: Any = None,
        parallel: bool = False,
    ) -> PipelineResult:
        """
        Run the evaluation pipeline.
        
        Args:
            model_endpoint: Model API endpoint
            model_interface: Pre-configured model interface
            test_cases: Test cases to evaluate
            parallel: Whether to run stages in parallel
            
        Returns:
            PipelineResult with all stage results
        """
        import time
        import uuid
        
        pipeline_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        logger.info(f"Starting pipeline run: {pipeline_id}")
        
        stage_results = {}
        stages_run = 0
        overall_status = "success"
        
        for stage in self._stages:
            if not stage.enabled:
                logger.info(f"Skipping disabled stage: {stage.name}")
                continue
            
            logger.info(f"Running stage: {stage.name}")
            
            try:
                # Call the appropriate method on the evaluator
                if hasattr(stage.evaluator, "evaluate_batch"):
                    result = stage.evaluator.evaluate_batch(
                        model_interface, test_cases
                    )
                elif hasattr(stage.evaluator, "run_evaluation"):
                    result = stage.evaluator.run_evaluation(
                        test_cases=test_cases
                    )
                elif hasattr(stage.evaluator, "detect_drift"):
                    result = stage.evaluator.detect_drift(test_cases)
                else:
                    result = {"status": "no_compatible_method"}
                
                stage_results[stage.name] = {
                    "status": "success",
                    "result": result,
                }
                stages_run += 1
                
            except Exception as e:
                logger.error(f"Error in stage {stage.name}: {e}")
                stage_results[stage.name] = {
                    "status": "error",
                    "error": str(e),
                }
                overall_status = "partial_failure"
        
        duration = time.time() - start_time
        
        logger.info(
            f"Pipeline {pipeline_id} completed in {duration:.2f}s. "
            f"Status: {overall_status}"
        )
        
        return PipelineResult(
            pipeline_id=pipeline_id,
            timestamp=datetime.now(),
            stages_run=stages_run,
            stage_results=stage_results,
            overall_status=overall_status,
            duration_seconds=duration,
        )
    
    def get_stages(self) -> list[PipelineStage]:
        """Get all pipeline stages."""
        return self._stages.copy()
    
    def __len__(self) -> int:
        """Return number of stages."""
        return len(self._stages)
    
    def __repr__(self) -> str:
        """String representation."""
        stage_names = [s.name for s in self._stages]
        return f"EvaluationPipeline(stages={stage_names})"
