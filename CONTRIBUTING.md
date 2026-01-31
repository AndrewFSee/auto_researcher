# Contributing to Auto Researcher

Thank you for your interest in contributing to Auto Researcher! This document provides guidelines and information for contributors.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/auto_researcher.git
   cd auto_researcher
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting (line-length: 100)
- **Ruff**: Linting and import sorting
- **mypy**: Static type checking

Run all checks before submitting:

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/ --fix

# Type check
mypy src/
```

## Testing

Run the test suite before submitting changes:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/auto_researcher --cov-report=html

# Run specific test categories
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/ -m "unit"      # Only unit tests
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run all checks**
   ```bash
   black src/ tests/
   ruff check src/ tests/
   mypy src/
   pytest tests/
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "feat: add new alpha model for XYZ"
   ```

5. **Push and create a PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or modifying tests
- `chore:` - Maintenance tasks

## Adding New Alpha Models

When adding a new alpha model:

1. Create the model in `src/auto_researcher/models/`
2. Follow the existing model interface pattern
3. Document the academic foundation and expected IC
4. Add unit tests in `tests/`
5. Run backtests to validate performance
6. Update README.md with model details

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## Questions?

Open an issue for questions or discussions about contributing.
