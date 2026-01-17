# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmarking suite comparing speed and accuracy of Python sentence tokenization libraries.

## Commands

```bash
# Install (use uv, not pip)
uv venv && uv sync                     # Core tokenizers only
uv sync --extra wtpsplit               # Include wtpsplit (PyTorch)
uv sync --extra wtpsplit-ort-gpu       # wtpsplit with ONNX Runtime GPU
uv sync --all-extras                   # All optional dependencies

# Run benchmarks
python benchmark.py                    # Run all (speed + edge cases)
python benchmark.py --speed            # Speed only
python benchmark.py --accuracy         # Edge case accuracy only
python benchmark.py --corpus           # Corpus evaluation (treebank + UD)
python benchmark.py --corpus-only ud   # Single corpus only
python benchmark.py --speed -n 5000 -t simple  # Custom: 5000 simple texts
python benchmark.py --speed --corpus --plot    # Generate PNG charts
```

## Architecture

Single-file benchmark (`benchmark.py`) with:
- Individual `benchmark_<library>()` functions returning `BenchmarkResult` dataclass
- Optional tokenizers (spaCy, stanza, wtpsplit) return `None` if not installed
- Speed benchmarks run on replicated test texts, edge case tests run on `EDGE_CASES` list
- Corpus evaluation uses NLTK treebank and Universal Dependencies English EWT (downloaded to `.cache/`)
- wtpsplit tests both `sat-1l-sm` (fastest) and `sat-3l-sm` (balanced) models

## Tokenizers

Core (always installed): BlingFire, NLTK, pySBD, syntok

Optional extras: `spacy`, `stanza`, `wtpsplit`, `wtpsplit-ort-cpu`, `wtpsplit-ort-gpu`, `all`
