# Contributing to LLM Behavior Evaluation Lab

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A GitHub account

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/llm-behavior-eval-lab.git
   cd llm-behavior-eval-lab
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

### Branching Strategy

- `main`: Stable release branch
- `develop`: Integration branch for features
- `feature/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates

### Making Changes

1. Create a new branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following our coding standards

3. Write or update tests for your changes

4. Run the test suite:
   ```bash
   pytest tests/ -v
   ```

5. Run linting and formatting:
   ```bash
   black src/
   ruff check src/ --fix
   mypy src/
   ```

6. Commit your changes with a clear message:
   ```bash
   git commit -m "feat: add new evaluation metric for X"
   ```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### Pull Request Process

1. Push your branch to your fork
2. Create a Pull Request against `develop`
3. Fill out the PR template
4. Wait for review and address feedback
5. Once approved, your PR will be merged

## Code Standards

### Python Style Guide

- Follow PEP 8
- Use type hints for all function signatures
- Maximum line length: 88 characters (Black default)
- Use docstrings for all public functions/classes

### Documentation

- Update README.md for significant changes
- Add docstrings following Google style
- Include usage examples for new features

### Testing

- Write tests for all new functionality
- Maintain >80% code coverage
- Use pytest fixtures for common setup

## Areas for Contribution

### High Priority

- [ ] Additional statistical tests for drift detection
- [ ] LLM-as-judge implementations
- [ ] Embedding-based similarity metrics
- [ ] Visualization components

### Good First Issues

- Adding more test cases to `examples/safety_test_cases.py`
- Improving documentation and examples
- Adding type hints to existing code
- Writing unit tests for edge cases

### Advanced Contributions

- New evaluator modules (e.g., hallucination detection)
- Performance optimizations
- Integration with popular ML frameworks
- API connectors for new model providers

## Testing Guidelines

### Unit Tests

```python
def test_function_name():
    """Test description."""
    # Arrange
    input_data = ...
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_value
```

### Integration Tests

Place in `tests/integration/` and mark with `@pytest.mark.integration`.

### Running Specific Tests

```bash
# Run a specific test file
pytest tests/test_evaluators.py -v

# Run tests matching a pattern
pytest -k "drift" -v

# Run with coverage
pytest --cov=src --cov-report=html
```

## Questions and Support

- Open an issue for bugs or feature requests
- Use discussions for questions
- Tag maintainers for urgent issues

## Code of Conduct

Please be respectful and constructive in all interactions. We follow the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to LLM Behavior Evaluation Lab! 🎉
