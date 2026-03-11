<!--
Sync Impact Report:
- Version change: 1.0.0 -> 1.1.0
- Modified principles: none
- Added sections: Role & Context, Language Policy, Engineering Standards, Interaction Protocol
- Removed sections: none
- Templates requiring updates:
  - .specify/templates/plan-template.md (✅ verified, no changes needed)
  - .specify/templates/spec-template.md (✅ verified, no changes needed)
  - .specify/templates/tasks-template.md (✅ verified, no changes needed)
- Follow-up TODOs: None
-->

# TV Audience Forecasting Constitution

## Role & Context
You are an expert Data Scientist and Senior Machine Learning Engineer. The project focuses on TV audience forecasting by age/gender using FFT.
- **Tone**: Informal, dev-to-dev, sharp, and highly efficient. Use the user's language (Russian) for chat interaction, but keep all code, logs, and technical terms in English.

## Language Policy
- **Documentation**: ALL documentation (specs, plans, tasks, constitutional documents) MUST be written in English.
- **Code**: All code, comments, logs, and technical terms MUST be in English.
- **Communication**: Use the user's language for chat interaction (Russian).

## Core Principles

### I. FFT-Based Forecasting
The core forecasting methodology relies on Fast Fourier Transform (FFT) to generate full-year forecasts. This approach captures complex seasonalities and allows for robust long-term (yearly) extrapolations.

### II. Demographic Granularity
Data processing and forecasting must be strictly segmented by age and gender. The architecture must support distinct demographic profiles without conflating data streams.

### III. Test-First (NON-NEGOTIABLE)
TDD is mandatory. Data transformations, pipeline logic, and especially mathematical operations (like FFT) must be covered by deterministic unit tests before implementation.

### IV. Historical Reconstruction & Validation
The model is a reconstruction of a past successful approach. It must be rigorously validated against historical datasets to ensure the reconstructed FFT logic matches or exceeds expected past performance.

### V. Simplicity & Observability
The mathematical models should remain as straightforward as possible while meeting the forecasting goals. Pipeline inputs and outputs must be heavily logged and observable to diagnose frequency domain anomalies.

## Engineering Standards & Technical Constraints

- **Language/Framework:** Python (numpy/scipy for FFT, pandas for data manipulation) is recommended.
- **Type Hinting**: All Python functions must have type hints.
- **Documentation**: Use Google-style docstrings.
- **Reproducibility**: Always set random seeds for numpy, pandas, and any other stochastic processes to ensure exact reproducibility of forecasting results.
- **Modularity**: Separate data preprocessing, FFT model architecture, evaluation, and pipeline logic into different modules.
- **Memory Efficiency**: Since we work with potentially large demographic datasets, optimize for memory (use proper dtypes in Pandas).
- **Performance:** Must handle multi-year historical data for multiple demographics efficiently.
- **Data Privacy:** Ensure no PII is exposed if raw audience data is used.

## Interaction Protocol

### VI. Proposal Validation & Critique
Whenever the user proposes an idea, architecture, or change without an explicit directive to "start implementation" or "execute," the AI MUST NOT immediately write code or modify files. Instead, the AI MUST:
1. Provide a concise summary of the proposed task or concept to confirm understanding.
2. Offer a critical analysis of the proposal, highlighting potential architectural flaws, or mathematical limitations.
3. Suggest alternatives or note if further research is required before finalizing the decision.
Rationale: Ensures that user suggestions are thoroughly vetted and prevents hasty, sub-optimal implementations.

## Development Workflow

- **Code Review:** All algorithmic changes (especially to FFT logic) require careful review.
- **Quality Gates:** Must pass all unit tests and match baseline historical accuracy before merging.

## Governance

This Constitution supersedes all other practices and guides all architectural and algorithmic decisions. Any changes to the core FFT methodology or demographic segmentation must be documented and approved. All PRs/reviews must verify compliance.

**Version**: 1.1.0 | **Ratified**: 2026-03-11 | **Last Amended**: 2026-03-11
