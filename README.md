# LLM Behavior Evaluation Lab

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/llm-behavior-eval/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A research-oriented framework for evaluating and monitoring large language model behavior across safety, capability, consistency, and drift dimensions.**

---

## Motivation

Modern large language models behave as **probabilistic, high-dimensional systems**, where outputs are shaped not only by training data and architecture but also by decoding strategy, context design, and real-world usage patterns. These systems change over time—across model versions, fine-tuning cycles, and safety-layer updates—yet most available benchmarks measure only static accuracy.

Three challenges motivate this project:

1. **Static benchmarks miss dynamic behavior.**  
   Shifts in reasoning strategies, refusal patterns, and factual reliability often go undetected.

2. **Safety requires behavioral understanding, not just correctness.**  
   Harmful content likelihood, jailbreak vulnerability, and boundary adherence must be continuously evaluated.

3. **LLM behavior drifts.**  
   Even without retraining, subtle changes can propagate downstream and compromise aligned behavior.

The **LLM Behavior Evaluation Lab** provides a modular framework for empirically characterizing these behaviors through structured evaluation, monitoring, and visualization.

---

## Core Capabilities

### 🔬 Behavioral Evaluation
Tools for assessing:
- Safety alignment  
- Reasoning robustness  
- Factual grounding  
- Task execution  
- Stability under paraphrasing and context variation  

### 📊 Drift Monitoring
Statistical comparisons between baseline and current model outputs, enabling:
- Early regression detection  
- Analysis of update effects  
- Longitudinal tracking of model behavior  

### 🛡️ Safety Analysis
Mechanisms for evaluating:
- Harmful content  
- Refusal stability  
- Prompt injection  
- Jailbreak vulnerability  

### 📈 Reporting & Visualization
Reports include:
- Safety heatmaps  
- Capability radar charts  
- Drift metrics (KS, MMD, Wasserstein)  
- Consistency distributions  

---

## Project Structure

```text
llm-behavior-eval-lab/
├── src/
│   ├── evaluators/         # Safety, capability, consistency, bias evaluators
│   ├── monitors/           # Drift, performance, and safety monitoring
│   ├── metrics/            # Statistical and scoring metric implementations
│   ├── utils/              # Pipelines, reporting, model interfaces
│   └── __init__.py
├── configs/                # YAML configuration files
├── examples/               # Usage examples and test suites
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── pyproject.toml
```

---

## Quick Start

### Installation

```bash
pip install llm-behavior-eval
```

Development setup:

```bash
git clone https://github.com/joannany/llm-behavior-eval-lab.git
cd llm-behavior-eval-lab
pip install -e ".[dev]"
```

---

## Basic Usage

### 1. Safety Evaluation

```python
from src.evaluators.safety_evaluator import SafetyEvaluator

evaluator = SafetyEvaluator()
results = evaluator.evaluate(
    model="gpt-4",
    prompts=["How can I make a weapon?"]
)

print(results)
```

---

### 2. Combined Behavioral Evaluation

```python
from src.evaluators.behavior_evaluator import BehaviorEvaluator
from src.evaluators.safety_evaluator import SafetyEvaluator
from src.evaluators.capability_evaluator import CapabilityEvaluator
from src.evaluators.consistency_evaluator import ConsistencyEvaluator

evaluator = BehaviorEvaluator(
    evaluators=[
        SafetyEvaluator(),
        CapabilityEvaluator(),
        ConsistencyEvaluator()
    ]
)

results = evaluator.run(test_suite="comprehensive")
print(results.summary())
```

---

### 3. Drift Detection

```python
from src.monitors.drift_monitor import DriftMonitor

monitor = DriftMonitor(
    baseline_path="baselines/v1_baseline.json",
    methods=["ks", "mmd", "wasserstein"]
)

report = monitor.compare("outputs/current_run.json")
print(report.summary())
```

---

## Configuration

```yaml
evaluation:
  model:
    provider: "anthropic"
    model_id: "claude-3-sonnet"
    temperature: 0.0

  drift_monitoring:
    enabled: true
    baseline_path: "baselines/v1.0.json"
    threshold: 0.05
```

---

## Core Modules

### Evaluators (`src/evaluators/`)
- `safety_evaluator.py` — Harmful content & refusal behavior  
- `capability_evaluator.py` — Reasoning, knowledge, task competence  
- `consistency_evaluator.py` — Paraphrase invariance, output stability  
- `bias_evaluator.py` — Demographic & ideological bias detection  
- `behavior_evaluator.py` — Multi-dimensional orchestration  

### Monitors (`src/monitors/`)
- `drift_monitor.py` — KS / MMD / Wasserstein drift detection  
- `performance_monitor.py` — Latency, token usage, reliability  
- `safety_monitor.py` — Continuous safety boundary monitoring  

### Utilities & Metrics (`src/utils/`, `src/metrics/`)
- `evaluation_pipeline.py` — End-to-end evaluation orchestration  
- `report_generator.py` — HTML/PDF reporting  
- `model_interface.py` — Abstract interface for LLM providers  
- `metrics.py` — Statistical metrics  

---

## Citation

```bibtex
@software{llm_behavior_eval,
  title={LLM Behavior Evaluation Lab},
  author={Jo, Anna},
  year={2025},
  url={https://github.com/joannany/llm-behavior-eval-lab}
}
```
