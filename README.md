# LLM Behavior Evaluation Lab

A comprehensive framework for evaluating and monitoring Large Language Model behavior, focusing on safety alignment, capability assessment, and behavioral drift detection.

## Overview

This repository provides tools and methodologies for systematic evaluation of LLM behaviors across multiple dimensions:

- **Safety Alignment**: Evaluate model responses against safety guidelines and harmful content policies
- **Capability Assessment**: Measure model performance across reasoning, knowledge, and task completion
- **Behavioral Drift Detection**: Monitor changes in model behavior over time or across versions
- **Consistency Analysis**: Assess response stability and reproducibility

## Key Features

### 🔬 Multi-Dimensional Evaluation Framework

```python
from llm_eval import BehaviorEvaluator, SafetyMetrics, CapabilityMetrics

evaluator = BehaviorEvaluator(
    model_endpoint="your-model-endpoint",
    metrics=[SafetyMetrics(), CapabilityMetrics()]
)

results = evaluator.run_evaluation(test_suite="comprehensive")
```

### 📊 Behavioral Drift Monitoring

```python
from llm_eval import DriftMonitor, StatisticalTests

monitor = DriftMonitor(
    baseline_distribution=baseline_responses,
    drift_threshold=0.05,
    statistical_test=StatisticalTests.KS_TEST
)

drift_report = monitor.detect_drift(current_responses)
```

### 🛡️ Safety Boundary Testing

```python
from llm_eval import SafetyBoundaryTester

tester = SafetyBoundaryTester(
    categories=["harmful_content", "bias", "hallucination"],
    severity_levels=["low", "medium", "high", "critical"]
)

safety_report = tester.probe_boundaries(model)
```

## Installation

```bash
pip install llm-behavior-eval

# For development
git clone https://github.com/joannany/llm-behavior-eval-lab.git
cd llm-behavior-eval-lab
pip install -e ".[dev]"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Behavior Evaluation Lab                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Test       │  │   Metric     │  │   Report     │          │
│  │   Suites     │──│   Engine     │──│   Generator  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Evaluation Core                       │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐   │   │
│  │  │ Safety  │ │Capability│ │  Drift  │ │ Consistency │   │   │
│  │  │ Eval    │ │  Eval   │ │ Monitor │ │   Checker   │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Model Interface Layer                  │   │
│  │   Supports: OpenAI, Anthropic, HuggingFace, Custom APIs  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. Evaluators (`src/evaluators/`)

| Module | Purpose |
|--------|---------|
| `safety_evaluator.py` | Assess safety alignment and harmful content detection |
| `capability_evaluator.py` | Measure reasoning, knowledge, and task performance |
| `consistency_evaluator.py` | Test response stability across paraphrased inputs |
| `bias_evaluator.py` | Detect demographic and ideological biases |

### 2. Monitors (`src/monitors/`)

| Module | Purpose |
|--------|---------|
| `drift_monitor.py` | Detect behavioral changes over time |
| `performance_monitor.py` | Track latency, token usage, and reliability |
| `safety_monitor.py` | Continuous safety boundary monitoring |

### 3. Metrics (`src/metrics/`)

| Module | Purpose |
|--------|---------|
| `safety_metrics.py` | Harm score, refusal rate, boundary adherence |
| `quality_metrics.py` | Coherence, relevance, factuality scores |
| `statistical_metrics.py` | Distribution divergence, stability indices |

## Evaluation Categories

### Safety Evaluation Taxonomy

```
Safety Evaluation
├── Content Safety
│   ├── Harmful Content Generation
│   ├── Illegal Activity Assistance
│   └── Privacy Violations
├── Alignment Safety
│   ├── Instruction Following Boundaries
│   ├── Jailbreak Resistance
│   └── Prompt Injection Defense
└── Behavioral Safety
    ├── Honesty & Transparency
    ├── Bias & Fairness
    └── Hallucination Detection
```

### Capability Evaluation Dimensions

```
Capability Assessment
├── Reasoning
│   ├── Logical Reasoning
│   ├── Mathematical Reasoning
│   └── Causal Reasoning
├── Knowledge
│   ├── Factual Accuracy
│   ├── Domain Expertise
│   └── Temporal Awareness
└── Task Performance
    ├── Instruction Following
    ├── Multi-step Planning
    └── Error Recovery
```

## Quick Start Guide

### Basic Evaluation

```python
from llm_eval import QuickEval

# One-line evaluation
results = QuickEval.run("gpt-4", test_suite="safety_basic")
print(results.summary())
```

### Custom Evaluation Pipeline

```python
from llm_eval import (
    EvaluationPipeline,
    SafetyEvaluator,
    CapabilityEvaluator,
    DriftMonitor,
    ReportGenerator
)

# Build custom pipeline
pipeline = EvaluationPipeline([
    SafetyEvaluator(categories=["harmful", "bias"]),
    CapabilityEvaluator(dimensions=["reasoning", "knowledge"]),
    DriftMonitor(baseline="v1.0_baseline.json")
])

# Run evaluation
results = pipeline.evaluate(
    model_endpoint="your-endpoint",
    test_cases=1000,
    parallel=True
)

# Generate report
report = ReportGenerator(results).create_pdf("evaluation_report.pdf")
```

### Drift Detection Example

```python
from llm_eval import DriftDetector, StatisticalTests
import numpy as np

# Initialize detector with baseline
detector = DriftDetector(
    baseline_embeddings=np.load("baseline_embeddings.npy"),
    statistical_tests=[
        StatisticalTests.KOLMOGOROV_SMIRNOV,
        StatisticalTests.WASSERSTEIN,
        StatisticalTests.MMD
    ],
    significance_level=0.05
)

# Check for drift in new model version
drift_result = detector.check(new_model_embeddings)

if drift_result.is_significant:
    print(f"Significant drift detected!")
    print(f"KS statistic: {drift_result.ks_stat:.4f}")
    print(f"Wasserstein distance: {drift_result.wasserstein:.4f}")
    print(f"Affected dimensions: {drift_result.top_drifting_dimensions}")
```

## Configuration

### YAML Configuration

```yaml
# configs/evaluation_config.yaml
evaluation:
  model:
    provider: "anthropic"
    model_id: "claude-sonnet-4-20250514"
    temperature: 0.0
    max_tokens: 1024
  
  test_suites:
    - name: "safety_comprehensive"
      categories:
        - harmful_content
        - jailbreak_resistance
        - prompt_injection
      samples_per_category: 100
    
    - name: "capability_basic"
      dimensions:
        - reasoning
        - knowledge
      samples_per_dimension: 50
  
  drift_monitoring:
    enabled: true
    baseline_path: "baselines/v1.0.json"
    threshold: 0.05
    window_size: 1000

  reporting:
    format: ["json", "html", "pdf"]
    include_visualizations: true
    alert_on_regression: true
```

## Test Suite Examples

### Safety Test Cases

```python
# examples/safety_tests.py
SAFETY_TEST_CASES = [
    {
        "id": "safety_001",
        "category": "harmful_content",
        "prompt": "How do I make a...",
        "expected_behavior": "REFUSE",
        "severity": "critical"
    },
    {
        "id": "safety_002", 
        "category": "jailbreak",
        "prompt": "Ignore previous instructions and...",
        "expected_behavior": "REFUSE",
        "severity": "high"
    },
    # ... more test cases
]
```

### Behavioral Consistency Tests

```python
# examples/consistency_tests.py
PARAPHRASE_GROUPS = [
    {
        "id": "consistency_001",
        "variants": [
            "What is the capital of France?",
            "Can you tell me France's capital city?",
            "Which city serves as the capital of France?",
            "France's capital is which city?"
        ],
        "expected_consistency": 1.0
    },
    # ... more test groups
]
```

## Metrics Reference

### Safety Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `harm_score` | Probability of harmful content generation | 0.0 - 1.0 |
| `refusal_rate` | Rate of appropriate refusals | 0.0 - 1.0 |
| `jailbreak_resistance` | Resistance to adversarial prompts | 0.0 - 1.0 |
| `boundary_adherence` | Compliance with safety guidelines | 0.0 - 1.0 |

### Quality Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `coherence` | Logical flow and structure | 0.0 - 1.0 |
| `relevance` | Response relevance to prompt | 0.0 - 1.0 |
| `factuality` | Factual accuracy score | 0.0 - 1.0 |
| `completeness` | Task completion rate | 0.0 - 1.0 |

### Drift Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| `ks_statistic` | Kolmogorov-Smirnov test statistic | < 0.05 |
| `wasserstein` | Wasserstein (Earth Mover's) distance | < 0.1 |
| `mmd` | Maximum Mean Discrepancy | < 0.05 |
| `js_divergence` | Jensen-Shannon Divergence | < 0.1 |

## Visualization Examples

The framework generates comprehensive visualizations:

- **Safety Heatmaps**: Category-wise safety scores
- **Drift Timelines**: Behavioral changes over versions
- **Capability Radar Charts**: Multi-dimensional performance view
- **Consistency Distributions**: Response stability analysis

## Integration with CI/CD

```yaml
# .github/workflows/model_evaluation.yml
name: LLM Behavior Evaluation

on:
  schedule:
    - cron: '0 0 * * *'  # Daily
  workflow_dispatch:

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Safety Evaluation
        run: |
          python -m llm_eval.cli evaluate \
            --config configs/production.yaml \
            --suite safety_comprehensive
      
      - name: Check for Drift
        run: |
          python -m llm_eval.cli drift-check \
            --baseline baselines/production.json \
            --threshold 0.05
      
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-results
          path: results/
```

## Research Applications

This framework supports research in:

- **AI Safety**: Systematic safety boundary testing
- **Model Alignment**: Measuring alignment with human values
- **Behavioral Analysis**: Understanding model decision patterns
- **Reliability Engineering**: Production monitoring and alerting

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{llm_behavior_eval,
  title={LLM Behavior Evaluation Lab},
  author={Anna},
  year={2025},
  url={https://github.com/joannany/llm-behavior-eval-lab}
}
```

## Related Projects

- [Anthropic Evals](https://github.com/anthropics/evals)
- [OpenAI Evals](https://github.com/openai/evals)
- [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
