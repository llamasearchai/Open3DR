# Contributing to Open3DR

Thank you for your interest in contributing to Open3DR! This document provides guidelines and information about contributing to our medical AI platform.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Docker (for containerized development)
- CUDA-compatible GPU (recommended)

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/Open3DR.git
   cd Open3DR
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

5. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Branching Strategy

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/feature-name` - New features
- `bugfix/bug-description` - Bug fixes
- `hotfix/critical-fix` - Critical production fixes

### Making Changes

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards
3. Write or update tests for your changes
4. Run the test suite:
   ```bash
   pytest
   ```

5. Ensure code quality:
   ```bash
   black open3d/
   isort open3d/
   flake8 open3d/
   mypy open3d/
   ```

6. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add new medical imaging feature

   - Implement advanced DICOM processing
   - Add support for 3D reconstruction
   - Include comprehensive tests
   - Update documentation"
   ```

7. Push to your fork and create a pull request

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 88)
- Use isort for import sorting
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes

### Medical Code Requirements

- All medical functionality must include comprehensive tests
- HIPAA compliance must be maintained in all patient data handling
- Medical calculations must be thoroughly validated
- Include appropriate error handling for medical edge cases

### Documentation

- Update documentation for any new features
- Include examples in docstrings
- Add type hints and parameter descriptions
- Update README.md if adding new functionality

## Testing

### Test Categories

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Medical Tests**: Validate medical accuracy and compliance
- **Performance Tests**: Benchmark performance-critical code

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration  
pytest -m medical
pytest -m performance

# Run with coverage
pytest --cov=open3d --cov-report=html
```

### Medical Testing Requirements

Medical functionality requires additional validation:

- Clinical accuracy validation
- HIPAA compliance verification
- Medical standard conformance
- Edge case handling for medical data

## Pull Request Process

1. Ensure your code passes all tests and quality checks
2. Update documentation as needed
3. Add a clear description of changes in the PR
4. Reference any related issues
5. Ensure medical functionality is properly validated
6. Wait for code review and address feedback

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Medical compliance is maintained
- [ ] Performance impact is considered
- [ ] Security implications are addressed

## Medical and Compliance Considerations

### HIPAA Compliance

- Never commit real patient data
- Use synthetic data for testing
- Ensure proper encryption for sensitive operations
- Maintain audit trails for all medical operations

### Medical Standards

- Follow HL7 FHIR standards for interoperability
- Ensure DICOM compliance for medical imaging
- Validate against FDA guidelines where applicable
- Maintain clinical accuracy in all medical algorithms

## Reporting Issues

When reporting issues:

1. Use the issue templates provided
2. Include steps to reproduce
3. Provide system information
4. For security issues, email nikjois@llamasearch.ai directly

### Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature request
- `documentation` - Documentation improvements
- `medical` - Medical functionality issues
- `security` - Security-related issues
- `performance` - Performance improvements

## Recognition

Contributors will be recognized in:

- Release notes for significant contributions
- Contributors section in README
- Academic publications (with permission)

## Questions?

For questions about contributing:

- Create a GitHub issue with the `question` label
- Email: nikjois@llamasearch.ai
- Check existing documentation and issues first

## License

By contributing to Open3DR, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Open3DR and helping advance medical AI technology! 