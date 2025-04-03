# Contributing to pytics

First off, thank you for considering contributing to pytics! It's people like you that make pytics such a great tool.

## Code of Conduct

While I don't have a formal code of conduct yet, I expect all contributors to:
- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a positive environment for everyone

## How to Contribute

There are many ways to contribute to pytics:
- Reporting bugs
- Suggesting enhancements
- Writing documentation
- Submitting code changes
- Helping others in discussions

### Reporting Bugs

If you find a bug, please report it using my [bug report template](https://github.com/HansMeershoek/pytics/issues/new?template=bug_report.md).

Before submitting a bug report:
1. Check if the bug has already been reported
2. Update to the latest version to see if the issue persists
3. Collect all necessary information about your environment

### Suggesting Enhancements

Have an idea for improving pytics? Submit a feature request using my [feature request template](https://github.com/HansMeershoek/pytics/issues/new?template=feature_request.md).

When suggesting enhancements:
1. Explain the problem you're trying to solve
2. Be specific about the solution you'd like
3. Consider the impact on existing features
4. Think about backward compatibility

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/pytics.git
   cd pytics
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Process

1. Create a new branch:
   ```bash
   git checkout -b feature-or-fix-name
   ```

2. Make your changes:
   - Follow the existing code style
   - Add tests for new features
   - Update documentation as needed

3. Run tests:
   ```bash
   pytest
   ```

4. Format your code:
   ```bash
   black .
   ```

5. Check types:
   ```bash
   mypy src/pytics
   ```

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation if you're adding or modifying features
3. Add or update tests as appropriate
4. Ensure all tests pass and code is formatted
5. Submit a pull request with a clear description:
   - What changes you made
   - Why you made them
   - Any special considerations

### Pull Request Guidelines

- Keep each PR focused on a single feature or fix
- Include tests for new features
- Update documentation as needed
- Follow the existing code style
- Write clear commit messages
- Respond to review comments promptly

## Documentation

- Update docstrings for any modified functions
- Update the README.md if adding new features
- Add examples for new functionality
- Keep documentation clear and concise

## Questions?

If you have questions about contributing, feel free to:
1. Open a discussion on GitHub
2. Ask in an existing relevant issue
3. Reach out to me

Thank you for contributing to pytics! 🎉 