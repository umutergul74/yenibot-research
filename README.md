# yeniBot Research

Phase 1 ML foundation for BTC/USDT perpetual futures direction modeling.

## Overview

This project builds a professional, bias-free ML pipeline that trains a binary TCN+GRU sequence model to identify BTC/USDT perpetual futures long opportunities from market microstructure features.

**Phase 1** ends at model validation. It does not include backtesting, trade execution, live deployment, or Phase 2 trading notebooks.

## Architecture

```
Binance Data (REST/Vision)
    │
    ▼
yenibot.data          ── Data Ingestion & Validation
    │
    ▼
yenibot.features      ── Feature Engineering (1H + 4H + 15m + Futures)
    │
    ▼
yenibot.labeling      ── Triple-Barrier Label Generation
    │
    ▼
yenibot.regime        ── HMM Regime Detection (Forward-Only)
    │
    ▼
yenibot.training      ── Purged Walk-Forward CV Training
    │
    ▼
yenibot.models        ── TCN+GRU Hybrid Encoder
    │
    ▼
yenibot.diagnostics   ── Metrics, Calibration, Reporting
    │
    ▼
yenibot.experiments   ── Experiment Orchestration & Profile Management
    │
    ▼
yenibot.automation    ── Auto-Review & Phase 2 Readiness Gates
```

## Project Structure

```
yenibot-research/
├── pyproject.toml              # Dependencies, tools (ruff, mypy, pytest)
├── Dockerfile                  # Reproducible environment
├── .github/workflows/ci.yml   # Lint + Type Check + Test pipeline
├── src/yenibot/
│   ├── config/                 # Configuration loading & profiles
│   │   ├── base.yaml           # Core settings (model, training, validation)
│   │   └── profiles/
│   │       ├── active.yaml     # Currently active feature profiles
│   │       └── archive.yaml    # Rejected/retired profiles with reasons
│   ├── data/                   # Binance data ingestion & validation
│   ├── features/               # Microstructure feature engineering
│   ├── labeling/               # Triple-barrier label generation
│   ├── models/                 # TCN+GRU hybrid encoder
│   ├── training/               # Walk-forward CV training pipeline
│   ├── regime/                 # HMM regime detection
│   ├── diagnostics/            # Metrics, calibration, reporting
│   │   └── reporting/          # Modular report generation
│   ├── experiments/            # Experiment orchestration (modular)
│   └── automation/             # Auto-review & readiness gates
├── tests/                      # Comprehensive test suite
├── notebooks/                  # Colab notebooks (01-05)
└── docs/                       # Architecture & operational docs
```

## Quick Start

### Installation

```bash
# Development install
pip install -e ".[dev]"

# With MLOps tools (MLflow, DVC)
pip install -e ".[dev,mlops]"
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/yenibot --cov-report=term-missing

# Skip slow tests
pytest -m "not slow"
```

### Linting & Type Checking

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/yenibot/
```

### Docker

```bash
# Build and run tests
docker build --target dev -t yenibot:dev .
docker run yenibot:dev

# Production
docker build --target prod -t yenibot:prod .
```

## Colab Workflow

Run notebooks in strict order:

1. `01_data_preparation.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_labeling.ipynb`
4. `04_training_walk_forward.ipynb`
5. `05_diagnostics_validation.ipynb`

## Phase 1 Readiness Gates

Phase 2 is blocked until all checks pass:

| Gate | Target | Status |
|------|--------|--------|
| Mean Rank IC | > 0.03 | ✅ |
| Rank IC Std | < 0.03 | ❌ Working |
| Positive IC Fraction | > 75% | ✅ |
| Long F1 | > 0.45 | ❌ Working |
| Future Unseen OOS | Ready | ❌ Waiting |

## Contributing

- Use focused commits with prefixes: `feat:`, `fix:`, `docs:`, `data:`, `model:`
- Run `ruff check` and `pytest` before committing
- Read `docs/SKILLS.md` before changing features, labels, or training

## License

MIT
