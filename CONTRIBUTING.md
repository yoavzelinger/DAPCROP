# Contributing to DAPCROP

Thank you for your interest in contributing to **DAPCROP**! We welcome contributions focused on domain adaptation, concept drift repair, and explainability for tree ensemble classifiers.

---

## Getting Started

1. **Fork the Repository**: Create a fork of `https://github.com/yoavzelinger/DAPCROP` on GitHub.
2. **Clone & Set Up**:
   ```bash
   git clone https://github.com/<your-username>/DAPCROP.git
   cd DAPCROP
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/my-ensemble-fixer
   ```

---

## Subtree Relationship with APPETITE

DAPCROP embeds core single-decision-tree components directly from **[APPETITE](https://github.com/yoavzelinger/APPETITE)** via git subtrees:
* [`Methodology/APPETITE/`](Methodology/APPETITE/)
* [`Evaluation/SingleModel/`](Evaluation/SingleModel/)

> [!IMPORTANT]
> If you are contributing changes to the base single-tree methodology or single-model evaluation pipelines, please submit your pull request directly to the **[APPETITE](https://github.com/yoavzelinger/APPETITE)** repository. Changes are then synchronized down into DAPCROP via the subtree workflow.
>
> If your changes focus on ensemble mapping (`MappedForest`), distributed tree diagnosis, forest fixer wrappers, or ensemble evaluation, please submit your pull request here in **DAPCROP**.

---

## Code Standards & Style

* **Python 3.10+ Typing**: Use PEP 604 union pipes (`A | B`) and PEP 585 built-in collections (`list`, `dict`, `set`, `tuple`).
* **Readable Naming**: Prefer descriptive, full names without ambiguous abbreviations.
* **Imports**: Use explicit absolute imports across packages.

---

## Submitting a Pull Request

1. **Verify Your Changes**:
   ```bash
   python -m Evaluation.SingleModel.evaluator --test iris
   ```
2. **Open a PR**: Open a pull request against the `main` branch with a clear description and benchmark results.
