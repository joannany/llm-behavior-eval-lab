"""
Tests for LLM Behavior Evaluation Lab
"""

import numpy as np
import sys
sys.path.insert(0, "/home/claude/llm-behavior-eval-lab/src")


class TestRefusalDetector:
    """Tests for RefusalDetector."""
    
    def test_detects_common_refusals(self):
        from evaluators.safety_evaluator import RefusalDetector
        detector = RefusalDetector()
        
        refusal_texts = [
            "I cannot help with that request.",
            "I'm sorry, but I can't assist with that.",
        ]
        
        for text in refusal_texts:
            detected, patterns = detector.detect(text)
            assert detected, f"Should detect refusal in: {text}"
    
    def test_does_not_flag_normal_text(self):
        from evaluators.safety_evaluator import RefusalDetector
        detector = RefusalDetector()
        
        detected, patterns = detector.detect("The capital of France is Paris.")
        assert not detected


class TestAnswerComparator:
    """Tests for AnswerComparator."""
    
    def test_exact_match(self):
        from evaluators.capability_evaluator import AnswerComparator
        comparator = AnswerComparator()
        score, explanation = comparator.compare("Paris", "Paris")
        assert score == 1.0
    
    def test_case_insensitive_match(self):
        from evaluators.capability_evaluator import AnswerComparator
        comparator = AnswerComparator()
        score, explanation = comparator.compare("paris", "Paris")
        assert score == 1.0


class TestStatisticalTests:
    """Tests for statistical drift detection."""
    
    def test_no_drift_identical_distributions(self):
        from monitors.drift_monitor import DriftDetector, StatisticalTests
        
        detector = DriftDetector(
            statistical_tests=[StatisticalTests.KOLMOGOROV_SMIRNOV],
            significance_level=0.05,
        )
        
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        detector.set_baseline(baseline)
        result = detector.check(current)
        
        assert result.overall_drift_score < 0.2


class TestMockInterface:
    """Tests for MockInterface."""
    
    def test_returns_default_response(self):
        from utils.model_interface import MockInterface
        mock = MockInterface()
        response = mock.generate("Any prompt")
        assert response == "Mock response for testing."
    
    def test_tracks_call_count(self):
        from utils.model_interface import MockInterface
        mock = MockInterface()
        mock.generate("Test 1")
        mock.generate("Test 2")
        assert mock.call_count == 2


class TestEvaluationPipeline:
    """Tests for EvaluationPipeline."""
    
    def test_initialization(self):
        from utils.evaluation_pipeline import EvaluationPipeline
        pipeline = EvaluationPipeline()
        assert len(pipeline) == 0
    
    def test_add_stage(self):
        from utils.evaluation_pipeline import EvaluationPipeline
        from evaluators.safety_evaluator import SafetyEvaluator
        
        pipeline = EvaluationPipeline()
        pipeline.add_stage("safety", SafetyEvaluator())
        
        assert len(pipeline) == 1


def run_tests():
    """Run all tests."""
    test_classes = [
        TestRefusalDetector,
        TestAnswerComparator,
        TestStatisticalTests,
        TestMockInterface,
        TestEvaluationPipeline,
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\nRunning: {test_class.__name__}")
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
