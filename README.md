# LLM Behavior Evaluation Lab

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/llm-behavior-eval/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-oriented framework for evaluating and monitoring large language model behavior across safety, capability, consistency, and drift dimensions.

---

## Motivation

Modern large language models behave as **probabilistic, high-dimensional systems**, where outputs are shaped not only by training data and architecture but also by decoding strategy, context design, and real-world usage patterns. These systems change over time—across model versions, fine-tuning cycles, and safety-layer updates—yet most available benchmarks measure only static accuracy.

Three challenges motivate this project:

1. **Static benchmarks miss dynamic behavior.**  
   Changes in reasoning style, refusal patterns, factual reliability, or tone often go undetected.

2. **Safety requires behavioral understanding, not just correctness.**  
   Harmful response likelihood, jailbreak vulnerability, and boundary adherence must be continuously evaluated.

3. **LLM behavior drifts.**  
   Even without retraining, subtle shifts can propagate downstream and degrade aligned behavior.

The LLM Behavior Evaluation Lab provides a modular framework for empirically characterizing these behaviors through structured evaluation, monitoring, and visualization.

---

## Core Capabilities

### 🔬 Behavioral Evaluation
Tools for assessing:
- safety alignment
- reasoning robustness
- factual grounding
- task execution
- response stability under paraphrasing and context variation

### 📊 Drift Monitoring
Statistical comparisons between baseline and current model outputs, enabling:
- early detection of regressions
- impact analysis of model updates
- longitudinal behavioral tracking

### 🛡️ Safety Analysis
Mechanisms for evaluating:
- harmful content generation
- refusal stability
- prompt-injection susceptibility
- jailbreak vulnerability

### 📈 Reporting & Visualization
HTML/PDF reports summarizing:
- safety heatmaps  
- capability radar charts  
- drift statistics (KS, MMD, Wasserstein)  
- consistency distributions  

---

## Project Structure

```
llm-behavior-eval-lab/
├── src/
│   ├── evaluators/         # Safety, capability, consistency, bias evaluators
│   ├── monitors/           # Drift, performance, and safety monitoring modules
│   ├── metrics/            # Statistical and scoring metric implementations
│   ├── utils/              # Pipelines, reporting tools, model interfaces
│   └── __init__.py
├── configs/                # YAML configuration files
├── examples/               # Usage examples and test cases
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── pyproject.toml
```

The design follows a **modular architecture**: evaluators measure behavior, monitors track changes, metrics quantify differences, and pipelines orchestrate execution.

---

## Quick Start

### Installation

```bash
pip install llm-behavior-eval
```

Development mode:

```bash
git clone https://github.com/joannany/llm-behavior-eval-lab.git
cd llm-behavior-eval-lab
pip install -e ".[dev]"
```

---

## Basic Usage

### 1. Run a Safety Evaluation

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

Behavior can be controlled via YAML:

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

This enables reproducible configuration for CI/CD pipelines or research workflows.

---

## Core Modules

### Evaluators (`src/evaluators/`)
- `safety_evaluator.py` — Harmful content detection, refusal stability  
- `capability_evaluator.py` — Reasoning, knowledge, task competence  
- `consistency_evaluator.py` — Paraphrase invariance, output stability  
- `bias_evaluator.py` — Demographic and ideological bias detection  
- `behavior_evaluator.py` — Multi-dimensional orchestration  

### Monitors (`src/monitors/`)
- `drift_monitor.py` — Behavioral drift detection using KS/MMD/Wasserstein  
- `performance_monitor.py` — Latency, token usage, reliability  
- `safety_monitor.py` — Continuous boundary monitoring  

### Utilities & Metrics (`src/utils/`, `src/metrics/`)
- `evaluation_pipeline.py` — End-to-end evaluation orchestration  
- `report_generator.py` — HTML/PDF reporting  
- `model_interface.py` — Abstract provider interface  
- `metrics.py` — Core statistical metrics  

---

## Example Evaluation Workflow

A typical evaluation workflow includes:

1. **Defining the target model**  
2. **Selecting evaluators** (safety, capability, consistency)  
3. **Running drift monitors** against historical baselines  
4. **Generating a report** for analysis or CI/CD gating  

Example:

```python
from src.utils.evaluation_pipeline import EvaluationPipeline
from src.evaluators import safety_evaluator, capability_evaluator
from src.monitors.drift_monitor import DriftMonitor

pipeline = EvaluationPipeline([
    safety_evaluator.SafetyEvaluator(),
    capability_evaluator.CapabilityEvaluator(),
    DriftMonitor(baseline_path="baselines/current.json")
])

results = pipeline.run(model="your-endpoint", test_cases=800)
results.to_report("reports/latest.html")
```

---

## Use Cases

- **AI Safety Research**  
  Evaluation of harmful content generation, refusal behavior, and safety drift.

- **Model Release Validation**  
  Detect regressions introduced by model updates.

- **LLM Product Quality Assurance**  
  Validate reasoning, factuality, and output consistency prior to deployment.

- **Research Experimentation**  
  Controlled studies on behavioral variation across architectures or prompts.

---

## Contributing

Contributions are welcome.  
Please see `docs/CONTRIBUTING.md` for guidelines.

---

## License

MIT License. See `LICENSE` for details.

---

## Citation

```bibtex
@software{llm_behavior_eval,
  title={LLM Behavior Evaluation Lab},
  author={Anna},
  year={2025},
  url={https://github.com/joannany/llm-behavior-eval-lab}
}
```
