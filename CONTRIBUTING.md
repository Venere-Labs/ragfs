# Contributing to RAGFS

Thank you for your interest in contributing to RAGFS! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Please read our full [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use a clear, descriptive title
3. Include:
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Rust version)
   - Relevant logs with `--verbose` flag

### Proposing Features

1. Open an issue describing the feature
2. Explain the use case and benefits
3. Discuss implementation approach
4. Wait for feedback before implementing

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Ensure tests pass
5. Submit a pull request

## Development Setup

### Prerequisites

- Rust 1.88 or later
- FUSE development libraries
- Git

### Building

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ragfs.git
cd ragfs

# Build all crates
cargo build

# Run tests
cargo test

# Build in release mode
cargo build --release
```

### Running Locally

```bash
# Run the CLI
cargo run --release -- index /path/to/test

# Run with verbose logging
cargo run --release -- -v index /path/to/test
```

## Code Style

### Formatting

All code must be formatted with `rustfmt`:

```bash
cargo fmt --all
```

### Linting

All code must pass `clippy` without warnings:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### Documentation

- All public items should have documentation comments
- Use `///` for item documentation
- Use `//!` for module documentation
- Include examples where helpful

### Testing

- Write tests for new functionality
- Run the full test suite before submitting:

```bash
cargo test --all
```

## Commit Messages

Follow conventional commit style:

```
type(scope): short description

Longer description if needed.
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code change that doesn't fix a bug or add a feature
- `test`: Adding or updating tests
- `chore`: Build, CI, or tooling changes

Examples:
```
feat(index): add support for PDF extraction
fix(query): handle empty result sets correctly
docs(readme): update installation instructions
```

## Project Structure

```
ragfs/
├── crates/
│   ├── ragfs/           # Main CLI binary
│   ├── ragfs-core/      # Core traits and types
│   ├── ragfs-fuse/      # FUSE filesystem
│   ├── ragfs-index/     # Indexing engine
│   ├── ragfs-chunker/   # Document chunking
│   ├── ragfs-embed/     # Embedding generation
│   ├── ragfs-extract/   # Content extraction
│   ├── ragfs-store/     # Vector storage
│   └── ragfs-query/     # Query execution
└── docs/                # Documentation
```

## Adding New Extractors

1. Create extractor in `ragfs-extract`
2. Implement the `ContentExtractor` trait
3. Register in `ExtractorRegistry`
4. Add tests
5. Update documentation

## Adding New Chunkers

1. Create chunker in `ragfs-chunker`
2. Implement the `Chunker` trait
3. Register in `ChunkerRegistry`
4. Add tests
5. Update documentation

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure CI passes
4. Request review from maintainers
5. Address review feedback
6. Squash commits if requested

## Questions?

Open an issue with the "question" label or reach out to maintainers.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT OR Apache-2.0).
