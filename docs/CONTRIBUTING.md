# Contributing to LLM Behavior Evaluation Lab

Thank you for considering a contribution! This project aims to build a robust, research-oriented framework for evaluating Large Language Model behavior. We welcome contributions of all kinds — new evaluators, drift metrics, documentation improvements, test cases, and more.

---

## 🛠️ Quick Start

### Requirements
- Python 3.10+
- Git

### Setup

```bash
git clone https://github.com/joannany/llm-behavior-eval-lab.git
cd llm-behavior-eval-lab
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

---

## 🔄 Workflow

### Branching
- **main** — stable releases  
- **develop** — integration branch (default PR target)  
- **feature/*** — new features  
- **fix/*** — bug fixes  
- **docs/*** — documentation updates  

### Development Steps
1. Create a new branch:
   ```bash
   git checkout develop
   git pull
   git checkout -b feature/your-feature
   ```
2. Make your changes.
3. Run quality checks:
   ```bash
   pytest -v
   ruff check src/ --fix
   mypy src/
   ```
4. Commit using Conventional Commit style:
   ```bash
   git commit -m "feat: add Wasserstein drift metric"
   ```

---

## 🧪 Testing

We use `pytest` and aim for reasonable coverage.

### Run tests
```bash
pytest -v
```

### Run tests by keyword
```bash
pytest -k "drift"
```

### Coverage
```bash
pytest --cov=src --cov-report=html
```

---

## 📏 Coding Guidelines

- Follow **PEP 8** and use **type hints**.
- All public classes/functions require docstrings.
- Use **Black** formatting (line length 88).
- Update documentation if you introduce or modify public APIs.

**Preferred docstring style: Google format.**

---

## 🗺️ Contribution Areas

### High-value contributions
- New statistical drift tests (e.g., Anderson–Darling)
- LLM-as-a-judge evaluators
- Embedding similarity drift metrics
- Visualization components

### Good first issues
- Add test cases to `examples/`
- Improve docstrings
- Type hints for utilities

---

## 🤝 Pull Requests

1. Push your branch.
2. Open a PR targeting `develop`.
3. Ensure all tests pass.
4. Respond to review comments promptly.

---

## 🌱 Code of Conduct

Please be respectful and constructive in all interactions.  
We follow the Contributor Covenant.

---

## ⚖️ License

By contributing, you agree that your work is released under the **MIT License**.

Thank you for helping improve the LLM Behavior Evaluation Lab! 🎉

