"""
Capability Evaluator

Evaluates LLM capabilities across reasoning, knowledge, and task performance
dimensions with comprehensive scoring and analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import logging
import json

import numpy as np

from .behavior_evaluator import (
    TestCase,
    EvaluationResult,
    MetricCalculator,
)

logger = logging.getLogger(__name__)


class CapabilityDimension(Enum):
    """Dimensions of capability evaluation."""
    LOGICAL_REASONING = "logical_reasoning"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    CAUSAL_REASONING = "causal_reasoning"
    ANALOGICAL_REASONING = "analogical_reasoning"
    FACTUAL_KNOWLEDGE = "factual_knowledge"
    DOMAIN_EXPERTISE = "domain_expertise"
    TEMPORAL_AWARENESS = "temporal_awareness"
    INSTRUCTION_FOLLOWING = "instruction_following"
    MULTI_STEP_PLANNING = "multi_step_planning"
    ERROR_RECOVERY = "error_recovery"
    CODE_GENERATION = "code_generation"
    CREATIVE_WRITING = "creative_writing"


class DifficultyLevel(Enum):
    """Difficulty levels for capability tests."""
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class CapabilityTestCase(TestCase):
    """Extended test case with capability-specific fields."""
    dimension: CapabilityDimension = CapabilityDimension.LOGICAL_REASONING
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    expected_answer: str = ""
    acceptable_answers: list[str] = field(default_factory=list)
    scoring_rubric: dict = field(default_factory=dict)
    requires_reasoning_chain: bool = False
    max_tokens: int = 1024


@dataclass
class CapabilityResult(EvaluationResult):
    """Extended result with capability-specific metrics."""
    dimension: CapabilityDimension | None = None
    difficulty: DifficultyLevel | None = None
    correctness_score: float = 0.0
    reasoning_score: float = 0.0
    completeness_score: float = 0.0
    format_adherence: float = 1.0
    extracted_answer: str = ""
    reasoning_chain: list[str] = field(default_factory=list)


class AnswerExtractor:
    """Extracts and normalizes answers from model responses."""
    
    def __init__(self, extraction_patterns: dict | None = None):
        self.patterns = extraction_patterns or {}
    
    def extract(
        self, 
        response: str, 
        dimension: CapabilityDimension
    ) -> tuple[str, float]:
        """
        Extract the answer from a response.
        
        Args:
            response: Full model response
            dimension: The capability dimension being tested
            
        Returns:
            Tuple of (extracted_answer, confidence_score)
        """
        # Simple extraction logic - look for common answer patterns
        response_lower = response.lower().strip()
        
        # Check for explicit answer markers
        answer_markers = [
            "the answer is",
            "answer:",
            "result:",
            "therefore,",
            "thus,",
            "so the answer is",
            "final answer:",
        ]
        
        for marker in answer_markers:
            if marker in response_lower:
                idx = response_lower.index(marker) + len(marker)
                # Extract until end of sentence or line
                remaining = response[idx:].strip()
                end_idx = min(
                    remaining.find(".") if remaining.find(".") > 0 else len(remaining),
                    remaining.find("\n") if remaining.find("\n") > 0 else len(remaining),
                )
                extracted = remaining[:end_idx].strip()
                if extracted:
                    return extracted, 0.9
        
        # Fallback: return last sentence or line
        lines = response.strip().split("\n")
        last_line = lines[-1].strip() if lines else ""
        
        return last_line, 0.5
    
    def normalize(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        # Remove common formatting
        normalized = answer.lower().strip()
        normalized = normalized.replace(",", "")
        normalized = normalized.rstrip(".")
        return normalized


class AnswerComparator:
    """Compares model answers against expected answers."""
    
    def __init__(
        self,
        exact_match_weight: float = 0.6,
        semantic_match_weight: float = 0.4,
    ):
        self.exact_match_weight = exact_match_weight
        self.semantic_match_weight = semantic_match_weight
        self.extractor = AnswerExtractor()
    
    def compare(
        self,
        model_answer: str,
        expected_answer: str,
        acceptable_answers: list[str] | None = None,
    ) -> tuple[float, str]:
        """
        Compare model answer to expected answer.
        
        Args:
            model_answer: The model's answer
            expected_answer: The correct answer
            acceptable_answers: Alternative acceptable answers
            
        Returns:
            Tuple of (score, explanation)
        """
        all_acceptable = [expected_answer]
        if acceptable_answers:
            all_acceptable.extend(acceptable_answers)
        
        # Normalize all answers
        model_normalized = self.extractor.normalize(model_answer)
        expected_normalized = [self.extractor.normalize(a) for a in all_acceptable]
        
        # Exact match check
        if model_normalized in expected_normalized:
            return 1.0, "Exact match with expected answer"
        
        # Partial/fuzzy matching
        best_similarity = 0.0
        best_match = ""
        
        for expected in expected_normalized:
            similarity = self._calculate_similarity(model_normalized, expected)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = expected
        
        if best_similarity >= 0.9:
            return 0.95, f"Near-exact match with '{best_match}'"
        elif best_similarity >= 0.7:
            return 0.7, f"Partial match with '{best_match}' (sim={best_similarity:.2f})"
        elif best_similarity >= 0.5:
            return 0.4, f"Weak match with '{best_match}' (sim={best_similarity:.2f})"
        
        return 0.0, f"No match found. Model: '{model_answer}', Expected: '{expected_answer}'"
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using Levenshtein ratio."""
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Simple character-based similarity
        common_chars = sum(1 for c in s1 if c in s2)
        return (2.0 * common_chars) / (len1 + len2)


class ReasoningChainAnalyzer:
    """Analyzes the quality of reasoning chains in responses."""
    
    REASONING_MARKERS = [
        "first,", "second,", "third,", "finally,",
        "step 1", "step 2", "step 3",
        "because", "therefore", "thus", "hence",
        "since", "given that", "it follows that",
        "we can see that", "this means", "this implies",
    ]
    
    def analyze(self, response: str) -> dict:
        """
        Analyze the reasoning chain in a response.
        
        Args:
            response: Full model response
            
        Returns:
            Analysis dictionary with reasoning metrics
        """
        response_lower = response.lower()
        
        # Count reasoning markers
        marker_count = sum(
            1 for marker in self.REASONING_MARKERS
            if marker in response_lower
        )
        
        # Estimate number of reasoning steps
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        sentences = response.replace("\n", " ").split(".")
        
        # Look for numbered or bulleted steps
        numbered_steps = sum(
            1 for line in lines
            if any(line.startswith(f"{i}") for i in range(1, 20))
        )
        
        # Calculate reasoning density
        word_count = len(response.split())
        reasoning_density = marker_count / max(word_count / 100, 1)
        
        return {
            "marker_count": marker_count,
            "estimated_steps": max(numbered_steps, marker_count // 2),
            "reasoning_density": reasoning_density,
            "has_clear_structure": numbered_steps > 0 or marker_count >= 3,
            "line_count": len(lines),
            "word_count": word_count,
        }
    
    def score_reasoning(self, response: str, requires_chain: bool = True) -> float:
        """
        Score the reasoning quality.
        
        Args:
            response: Full model response
            requires_chain: Whether explicit reasoning chain is required
            
        Returns:
            Reasoning quality score (0-1)
        """
        analysis = self.analyze(response)
        
        if not requires_chain:
            # Just check for some reasoning markers
            if analysis["marker_count"] >= 1:
                return 0.8
            return 0.5
        
        # Score based on reasoning quality
        score = 0.0
        
        # Clear structure bonus
        if analysis["has_clear_structure"]:
            score += 0.4
        
        # Multiple steps bonus
        if analysis["estimated_steps"] >= 3:
            score += 0.3
        elif analysis["estimated_steps"] >= 1:
            score += 0.15
        
        # Reasoning density bonus
        if analysis["reasoning_density"] >= 0.5:
            score += 0.3
        elif analysis["reasoning_density"] >= 0.2:
            score += 0.15
        
        return min(1.0, score)


class CapabilityEvaluator:
    """
    Evaluator for LLM capability assessment.
    
    Measures model performance across reasoning, knowledge, and task
    completion dimensions with detailed scoring and analysis.
    
    Example:
        >>> evaluator = CapabilityEvaluator(
        ...     dimensions=[CapabilityDimension.LOGICAL_REASONING],
        ...     use_llm_judge=True
        ... )
        >>> results = evaluator.evaluate(model, test_cases)
    """
    
    def __init__(
        self,
        dimensions: list[CapabilityDimension] | None = None,
        difficulty_filter: DifficultyLevel | None = None,
        answer_comparator: AnswerComparator | None = None,
        reasoning_analyzer: ReasoningChainAnalyzer | None = None,
        use_llm_judge: bool = False,
        llm_judge: Any = None,
        config: dict | None = None,
    ):
        """
        Initialize the capability evaluator.
        
        Args:
            dimensions: Capability dimensions to evaluate
            difficulty_filter: Minimum difficulty level to include
            answer_comparator: Custom answer comparison instance
            reasoning_analyzer: Custom reasoning analysis instance
            use_llm_judge: Whether to use LLM-as-judge for evaluation
            llm_judge: LLM instance for judging
            config: Additional configuration
        """
        self.dimensions = dimensions or list(CapabilityDimension)
        self.difficulty_filter = difficulty_filter
        self.answer_comparator = answer_comparator or AnswerComparator()
        self.reasoning_analyzer = reasoning_analyzer or ReasoningChainAnalyzer()
        self.use_llm_judge = use_llm_judge
        self.llm_judge = llm_judge
        self.config = config or {}
        self.answer_extractor = AnswerExtractor()
        
        logger.info(
            f"Initialized CapabilityEvaluator with {len(self.dimensions)} dimensions"
        )
    
    def evaluate_response(
        self,
        test_case: CapabilityTestCase,
        response: str,
        latency_ms: float = 0.0,
    ) -> CapabilityResult:
        """
        Evaluate a single response for capability metrics.
        
        Args:
            test_case: The capability test case
            response: Model response to evaluate
            latency_ms: Response latency in milliseconds
            
        Returns:
            CapabilityResult with detailed metrics
        """
        # Extract answer
        extracted_answer, extraction_confidence = self.answer_extractor.extract(
            response, test_case.dimension
        )
        
        # Compare answers
        correctness_score, comparison_explanation = self.answer_comparator.compare(
            extracted_answer,
            test_case.expected_answer,
            test_case.acceptable_answers,
        )
        
        # Analyze reasoning
        reasoning_score = self.reasoning_analyzer.score_reasoning(
            response, test_case.requires_reasoning_chain
        )
        
        # Calculate completeness
        completeness_score = self._calculate_completeness(test_case, response)
        
        # Check format adherence
        format_adherence = self._check_format(test_case, response)
        
        # Calculate overall score
        weights = {
            "correctness": 0.5,
            "reasoning": 0.25,
            "completeness": 0.15,
            "format": 0.1,
        }
        
        overall_score = (
            correctness_score * weights["correctness"] +
            reasoning_score * weights["reasoning"] +
            completeness_score * weights["completeness"] +
            format_adherence * weights["format"]
        )
        
        # Determine pass/fail
        passed = correctness_score >= 0.7 and overall_score >= 0.5
        
        # Generate explanation
        explanation = (
            f"Correctness: {correctness_score:.2f} ({comparison_explanation}). "
            f"Reasoning: {reasoning_score:.2f}. "
            f"Completeness: {completeness_score:.2f}. "
            f"Format: {format_adherence:.2f}"
        )
        
        return CapabilityResult(
            test_case_id=test_case.id,
            passed=passed,
            score=overall_score,
            response=response,
            latency_ms=latency_ms,
            explanation=explanation,
            dimension=test_case.dimension,
            difficulty=test_case.difficulty,
            correctness_score=correctness_score,
            reasoning_score=reasoning_score,
            completeness_score=completeness_score,
            format_adherence=format_adherence,
            extracted_answer=extracted_answer,
        )
    
    def _calculate_completeness(
        self, 
        test_case: CapabilityTestCase, 
        response: str
    ) -> float:
        """Calculate how complete the response is."""
        if not response.strip():
            return 0.0
        
        # Basic heuristics
        word_count = len(response.split())
        
        # Very short responses are likely incomplete
        if word_count < 5:
            return 0.3
        elif word_count < 20:
            return 0.6
        
        # Check if response seems truncated
        if response.rstrip()[-1] not in ".!?\"')":
            return 0.7
        
        return 1.0
    
    def _check_format(
        self, 
        test_case: CapabilityTestCase, 
        response: str
    ) -> float:
        """Check if response adheres to expected format."""
        if not test_case.scoring_rubric:
            return 1.0
        
        format_requirements = test_case.scoring_rubric.get("format", {})
        if not format_requirements:
            return 1.0
        
        score = 1.0
        
        # Check various format requirements
        if format_requirements.get("requires_json"):
            try:
                json.loads(response)
            except json.JSONDecodeError:
                score -= 0.5
        
        if format_requirements.get("requires_bullet_points"):
            if not any(line.strip().startswith(("-", "*", "•")) 
                      for line in response.split("\n")):
                score -= 0.3
        
        if format_requirements.get("max_length"):
            if len(response) > format_requirements["max_length"]:
                score -= 0.2
        
        return max(0.0, score)
    
    def evaluate_batch(
        self,
        model_interface: Any,
        test_cases: list[CapabilityTestCase],
        parallel: bool = False,
    ) -> list[CapabilityResult]:
        """
        Evaluate a batch of capability test cases.
        
        Args:
            model_interface: Interface to query the model
            test_cases: List of capability test cases
            parallel: Whether to run in parallel
            
        Returns:
            List of CapabilityResult objects
        """
        import time
        
        results = []
        
        for test_case in test_cases:
            # Apply filters
            if test_case.dimension not in self.dimensions:
                continue
            
            if self.difficulty_filter:
                difficulty_order = list(DifficultyLevel)
                if difficulty_order.index(test_case.difficulty) < \
                   difficulty_order.index(self.difficulty_filter):
                    continue
            
            # Get model response
            start_time = time.time()
            try:
                response = model_interface.generate(
                    test_case.prompt,
                    max_tokens=test_case.max_tokens
                )
                latency_ms = (time.time() - start_time) * 1000
            except Exception as e:
                logger.error(f"Error generating response for {test_case.id}: {e}")
                response = ""
                latency_ms = 0.0
            
            # Evaluate response
            result = self.evaluate_response(test_case, response, latency_ms)
            results.append(result)
        
        return results
    
    def generate_report(self, results: list[CapabilityResult]) -> dict:
        """
        Generate a comprehensive capability report.
        
        Args:
            results: List of capability evaluation results
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not results:
            return {"error": "No results to report"}
        
        report = {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "pass_rate": sum(1 for r in results if r.passed) / len(results),
            "mean_overall_score": np.mean([r.score for r in results]),
            "mean_correctness": np.mean([r.correctness_score for r in results]),
            "mean_reasoning": np.mean([r.reasoning_score for r in results]),
            "mean_completeness": np.mean([r.completeness_score for r in results]),
            "mean_latency_ms": np.mean([r.latency_ms for r in results]),
            "by_dimension": {},
            "by_difficulty": {},
        }
        
        # Dimension breakdown
        by_dimension: dict[str, list[CapabilityResult]] = {}
        for result in results:
            dim = result.dimension.value if result.dimension else "unknown"
            if dim not in by_dimension:
                by_dimension[dim] = []
            by_dimension[dim].append(result)
        
        for dimension, dim_results in by_dimension.items():
            report["by_dimension"][dimension] = {
                "count": len(dim_results),
                "pass_rate": sum(1 for r in dim_results if r.passed) / len(dim_results),
                "mean_score": np.mean([r.score for r in dim_results]),
                "mean_correctness": np.mean([r.correctness_score for r in dim_results]),
            }
        
        # Difficulty breakdown
        by_difficulty: dict[str, list[CapabilityResult]] = {}
        for result in results:
            diff = result.difficulty.value if result.difficulty else "unknown"
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(result)
        
        for difficulty, diff_results in by_difficulty.items():
            report["by_difficulty"][difficulty] = {
                "count": len(diff_results),
                "pass_rate": sum(1 for r in diff_results if r.passed) / len(diff_results),
                "mean_score": np.mean([r.score for r in diff_results]),
            }
        
        return report
