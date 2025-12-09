"""
Consistency Evaluator

Evaluates response consistency and stability across paraphrased inputs,
repeated queries, and varying conditions.
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import logging

import numpy as np

from .behavior_evaluator import TestCase, EvaluationResult

logger = logging.getLogger(__name__)


class ConsistencyType(Enum):
    """Types of consistency evaluation."""
    PARAPHRASE = "paraphrase"
    SEMANTIC = "semantic"
    FACTUAL = "factual"
    STYLE = "style"
    TEMPORAL = "temporal"


@dataclass
class ParaphraseGroup:
    """A group of semantically equivalent prompts."""
    id: str
    base_prompt: str
    variants: list[str]
    expected_consistency: float = 1.0
    topic: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ConsistencyResult(EvaluationResult):
    """Result of consistency evaluation."""
    consistency_score: float = 0.0
    semantic_similarity: float = 0.0
    factual_agreement: float = 0.0
    response_variance: float = 0.0
    outlier_responses: list[int] = field(default_factory=list)
    pairwise_similarities: list[float] = field(default_factory=list)


class SemanticSimilarityCalculator:
    """Calculates semantic similarity between responses."""
    
    def __init__(self, method: str = "simple"):
        """
        Initialize similarity calculator.
        
        Args:
            method: Similarity method ("simple", "embedding", "llm")
        """
        self.method = method
    
    def calculate(self, response1: str, response2: str) -> float:
        """
        Calculate semantic similarity between two responses.
        
        Args:
            response1: First response
            response2: Second response
            
        Returns:
            Similarity score (0-1)
        """
        if self.method == "simple":
            return self._simple_similarity(response1, response2)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")
    
    def _simple_similarity(self, r1: str, r2: str) -> float:
        """Calculate simple word-based similarity."""
        if not r1 or not r2:
            return 0.0 if (not r1 and not r2) == False else 1.0
        
        words1 = set(r1.lower().split())
        words2 = set(r2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        # Jaccard similarity
        return len(intersection) / len(union)
    
    def calculate_pairwise(self, responses: list[str]) -> list[float]:
        """
        Calculate all pairwise similarities.
        
        Args:
            responses: List of responses to compare
            
        Returns:
            List of pairwise similarity scores
        """
        similarities = []
        n = len(responses)
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.calculate(responses[i], responses[j])
                similarities.append(sim)
        
        return similarities


class FactualAgreementChecker:
    """Checks factual agreement across multiple responses."""
    
    def __init__(self, entity_extractor: Any = None):
        self.entity_extractor = entity_extractor
    
    def check_agreement(self, responses: list[str]) -> tuple[float, dict]:
        """
        Check factual agreement across responses.
        
        Args:
            responses: List of responses to check
            
        Returns:
            Tuple of (agreement_score, disagreement_details)
        """
        if len(responses) < 2:
            return 1.0, {}
        
        # Simple approach: extract numbers and compare
        numbers_per_response = []
        for response in responses:
            numbers = self._extract_numbers(response)
            numbers_per_response.append(numbers)
        
        # Check if all responses have same numbers
        if all(len(nums) == 0 for nums in numbers_per_response):
            return 1.0, {"note": "No numbers found to compare"}
        
        # Compare numbers across responses
        all_numbers = [frozenset(nums) for nums in numbers_per_response]
        unique_sets = set(all_numbers)
        
        agreement = 1.0 - (len(unique_sets) - 1) / len(responses)
        
        disagreements = {}
        if len(unique_sets) > 1:
            disagreements["number_variations"] = [
                list(nums) for nums in unique_sets
            ]
        
        return agreement, disagreements
    
    def _extract_numbers(self, text: str) -> list[float]:
        """Extract numbers from text."""
        import re
        pattern = r'\b\d+(?:\.\d+)?\b'
        matches = re.findall(pattern, text)
        return [float(m) for m in matches]


class ConsistencyEvaluator:
    """
    Evaluator for response consistency assessment.
    
    Tests how consistently a model responds to semantically equivalent
    prompts and identifies instabilities in model behavior.
    
    Example:
        >>> evaluator = ConsistencyEvaluator(
        ...     similarity_threshold=0.8,
        ...     min_consistency=0.9
        ... )
        >>> results = evaluator.evaluate_paraphrase_group(model, group)
    """
    
    def __init__(
        self,
        similarity_calculator: SemanticSimilarityCalculator | None = None,
        agreement_checker: FactualAgreementChecker | None = None,
        similarity_threshold: float = 0.7,
        min_consistency: float = 0.8,
        config: dict | None = None,
    ):
        """
        Initialize consistency evaluator.
        
        Args:
            similarity_calculator: Custom similarity calculator
            agreement_checker: Custom factual agreement checker
            similarity_threshold: Minimum similarity for consistency
            min_consistency: Minimum consistency score to pass
            config: Additional configuration
        """
        self.similarity_calculator = (
            similarity_calculator or SemanticSimilarityCalculator()
        )
        self.agreement_checker = (
            agreement_checker or FactualAgreementChecker()
        )
        self.similarity_threshold = similarity_threshold
        self.min_consistency = min_consistency
        self.config = config or {}
        
        logger.info(
            f"Initialized ConsistencyEvaluator with threshold={similarity_threshold}"
        )
    
    def evaluate_paraphrase_group(
        self,
        model_interface: Any,
        group: ParaphraseGroup,
    ) -> ConsistencyResult:
        """
        Evaluate consistency across a paraphrase group.
        
        Args:
            model_interface: Interface to query the model
            group: ParaphraseGroup with prompt variants
            
        Returns:
            ConsistencyResult with detailed metrics
        """
        import time
        
        # Get responses for all variants
        responses = []
        latencies = []
        all_prompts = [group.base_prompt] + group.variants
        
        for prompt in all_prompts:
            start = time.time()
            try:
                response = model_interface.generate(prompt)
                latency = (time.time() - start) * 1000
            except Exception as e:
                logger.error(f"Error getting response: {e}")
                response = ""
                latency = 0.0
            
            responses.append(response)
            latencies.append(latency)
        
        # Calculate pairwise similarities
        pairwise_sims = self.similarity_calculator.calculate_pairwise(responses)
        
        # Calculate overall consistency score
        if pairwise_sims:
            mean_similarity = np.mean(pairwise_sims)
            min_similarity = np.min(pairwise_sims)
            similarity_variance = np.var(pairwise_sims)
        else:
            mean_similarity = 1.0
            min_similarity = 1.0
            similarity_variance = 0.0
        
        # Check factual agreement
        factual_agreement, disagreements = self.agreement_checker.check_agreement(
            responses
        )
        
        # Identify outliers
        outliers = self._identify_outliers(responses, pairwise_sims)
        
        # Calculate overall consistency
        consistency_score = (
            0.5 * mean_similarity +
            0.3 * factual_agreement +
            0.2 * (1.0 - similarity_variance)
        )
        
        # Determine pass/fail
        passed = (
            consistency_score >= self.min_consistency and
            min_similarity >= self.similarity_threshold * 0.8
        )
        
        # Build explanation
        explanation = (
            f"Mean similarity: {mean_similarity:.3f}. "
            f"Factual agreement: {factual_agreement:.3f}. "
            f"Variance: {similarity_variance:.4f}. "
            f"Outliers: {len(outliers)}"
        )
        
        return ConsistencyResult(
            test_case_id=group.id,
            passed=passed,
            score=consistency_score,
            response=responses[0] if responses else "",
            latency_ms=np.mean(latencies) if latencies else 0.0,
            explanation=explanation,
            consistency_score=consistency_score,
            semantic_similarity=mean_similarity,
            factual_agreement=factual_agreement,
            response_variance=similarity_variance,
            outlier_responses=outliers,
            pairwise_similarities=pairwise_sims,
        )
    
    def _identify_outliers(
        self, 
        responses: list[str], 
        pairwise_sims: list[float]
    ) -> list[int]:
        """Identify outlier responses."""
        if len(responses) < 3:
            return []
        
        # Calculate mean similarity to all other responses for each response
        n = len(responses)
        mean_sims = []
        
        idx = 0
        for i in range(n):
            sims_for_i = []
            for j in range(n):
                if i != j:
                    if i < j:
                        sims_for_i.append(pairwise_sims[idx])
                        idx += 1
                    else:
                        # Find the corresponding index
                        pair_idx = sum(range(n - 1, n - 1 - j, -1)) + (i - j - 1)
                        sims_for_i.append(pairwise_sims[pair_idx] if pair_idx < len(pairwise_sims) else 0.5)
            mean_sims.append(np.mean(sims_for_i) if sims_for_i else 1.0)
        
        # Identify responses with low mean similarity
        threshold = np.mean(mean_sims) - 1.5 * np.std(mean_sims)
        outliers = [i for i, sim in enumerate(mean_sims) if sim < threshold]
        
        return outliers
    
    def evaluate_batch(
        self,
        model_interface: Any,
        groups: list[ParaphraseGroup],
    ) -> list[ConsistencyResult]:
        """
        Evaluate consistency across multiple paraphrase groups.
        
        Args:
            model_interface: Interface to query the model
            groups: List of paraphrase groups
            
        Returns:
            List of ConsistencyResult objects
        """
        results = []
        for group in groups:
            result = self.evaluate_paraphrase_group(model_interface, group)
            results.append(result)
        return results
    
    def generate_report(self, results: list[ConsistencyResult]) -> dict:
        """
        Generate a consistency evaluation report.
        
        Args:
            results: List of consistency results
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not results:
            return {"error": "No results to report"}
        
        return {
            "total_groups": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "pass_rate": sum(1 for r in results if r.passed) / len(results),
            "mean_consistency": np.mean([r.consistency_score for r in results]),
            "mean_semantic_similarity": np.mean([r.semantic_similarity for r in results]),
            "mean_factual_agreement": np.mean([r.factual_agreement for r in results]),
            "total_outliers": sum(len(r.outlier_responses) for r in results),
            "groups_with_outliers": sum(
                1 for r in results if len(r.outlier_responses) > 0
            ),
        }
