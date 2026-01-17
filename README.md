# Sentence Tokenizer Benchmarks

Benchmarks comparing speed and accuracy of Python sentence tokenization libraries.

## Libraries Tested

| Library | Type | Notes |
|---------|------|-------|
| [BlingFire](https://github.com/microsoft/BlingFire) | Rule-based (C++) | Microsoft, fast |
| [NLTK](https://www.nltk.org/) | Rule-based (Python) | Punkt tokenizer |
| [pySBD](https://github.com/nipunsadvilkar/pySBD) | Rule-based (Python) | 22 languages |
| [syntok](https://github.com/fnl/syntok) | Rule-based (Python) | Also does word tokenization |
| [wtpsplit](https://github.com/segment-any-text/wtpsplit) | ML-based | SOTA accuracy, multilingual |

## Installation

```bash
# Core tokenizers only
uv pip install -e .

# Include wtpsplit (PyTorch)
uv pip install -e ".[wtpsplit]"

# Include wtpsplit with ONNX Runtime (faster)
uv pip install -e ".[wtpsplit-ort-gpu]"  # GPU
uv pip install -e ".[wtpsplit-ort-cpu]"  # CPU only
```

## Usage

```bash
# Run all benchmarks (speed + edge case accuracy)
python benchmark.py

# Speed only
python benchmark.py --speed

# Edge case accuracy only
python benchmark.py --accuracy

# Corpus evaluation (NLTK treebank + UD English)
python benchmark.py --corpus

# Single corpus evaluation
python benchmark.py --corpus-only treebank
python benchmark.py --corpus-only ud

# Customize speed benchmark
python benchmark.py --speed -n 5000 -t simple
```

## Results

### Speed (1000 complex texts)

| Library | Time | Per text | vs BlingFire |
|---------|------|----------|--------------|
| **BlingFire** | 0.06s | 0.06ms | 1.0x |
| NLTK | 0.32s | 0.32ms | 5.2x slower |
| syntok | 0.39s | 0.39ms | 6.4x slower |
| pySBD | 1.21s | 1.21ms | 20x slower |
| wtpsplit (ORT GPU) | ~0.73s | ~0.73ms | ~12x slower |
| wtpsplit (PyTorch GPU) | ~1.04s | ~1.04ms | ~17x slower |
| wtpsplit (CPU) | ~27s | ~27ms | ~450x slower |

### Accuracy (Edge Cases)

Test: `"Dr. Smith went to Washington D.C. on Jan. 5th. He met with Sen. Johnson at 3 p.m."`

| Library | Result | Correct? |
|---------|--------|----------|
| BlingFire | `["Dr. Smith went to Washington D.C. on Jan. 5th.", "He met with Sen. Johnson at 3 p.m."]` | ✅ |
| NLTK | `["Dr. Smith went to Washington D.C. on Jan. 5th.", "He met with Sen. Johnson at 3 p.m."]` | ✅ |
| pySBD | `["Dr. Smith went to Washington D.C. on Jan. 5th.", "He met with Sen. Johnson at 3 p.m."]` | ✅ |
| syntok | `["Dr. Smith went to Washington D.C. on Jan.", "5th.", "He met with Sen. Johnson at 3 p.m."]` | ❌ |
| wtpsplit | `["Dr. Smith went to Washington D.C. on Jan. 5th.", "He met with Sen. Johnson at 3 p.m."]` | ✅ |

## Recommendations

- **Best overall**: BlingFire (fastest + accurate)
- **Best fallback**: NLTK (pure Python, no binary deps, good accuracy)
- **Best accuracy**: wtpsplit (ML-based, but much slower)
- **Avoid**: syntok (fails on common abbreviation patterns)

## License

MIT
