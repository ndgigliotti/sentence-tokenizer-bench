#!/usr/bin/env python3
"""Benchmark sentence tokenization libraries for speed and accuracy."""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

# Test texts
SIMPLE_TEXT = "This is a sentence. This is another sentence."

COMPLEX_TEXT = """Dr. Smith went to Washington D.C. on Jan. 5th. He met with Sen. Johnson at 3 p.m. The meeting was productive! They discussed the U.S. economy. "This is great," he said. Meanwhile, Prof. Jones (Ph.D.) published a new paper. It cost $1.5 million to fund. What do you think? I think it's worth it... The results were published in Nature vol. 123, pp. 45-67."""

EDGE_CASES = [
    ("Abbreviations", "Dr. Smith and Prof. Jones met at 3 p.m. in Washington D.C."),
    ("Dates", "The event is on Jan. 5th. Please arrive early."),
    ("Numbers", "It cost $1.5 million. The ROI was 2.5x."),
    ("Quotes", '"This is great," he said. "I agree!" she replied.'),
    ("Ellipsis", "I think it's worth it... The results were good."),
    ("URLs", "Visit https://example.com. It has great info."),
    ("Initials", "J. K. Rowling wrote Harry Potter. It sold millions."),
]


@dataclass
class BenchmarkResult:
    name: str
    time_seconds: float
    num_texts: int
    sentences: list[str] | None = None

    @property
    def ms_per_text(self) -> float:
        return (self.time_seconds / self.num_texts) * 1000


def benchmark_blingfire(texts: list[str]) -> BenchmarkResult:
    import blingfire as bf

    start = time.perf_counter()
    for t in texts:
        bf.text_to_sentences_and_offsets(t)
    elapsed = time.perf_counter() - start

    # Get sentences for accuracy check
    sentences = bf.text_to_sentences(texts[0]).split("\n")
    return BenchmarkResult("BlingFire", elapsed, len(texts), sentences)


def benchmark_nltk(texts: list[str]) -> BenchmarkResult:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    start = time.perf_counter()
    for t in texts:
        nltk.sent_tokenize(t)
    elapsed = time.perf_counter() - start

    sentences = nltk.sent_tokenize(texts[0])
    return BenchmarkResult("NLTK", elapsed, len(texts), sentences)


def benchmark_pysbd(texts: list[str]) -> BenchmarkResult:
    import pysbd

    segmenter = pysbd.Segmenter(language="en", clean=False)

    start = time.perf_counter()
    for t in texts:
        segmenter.segment(t)
    elapsed = time.perf_counter() - start

    sentences = segmenter.segment(texts[0])
    return BenchmarkResult("pySBD", elapsed, len(texts), sentences)


def benchmark_syntok(texts: list[str]) -> BenchmarkResult:
    from syntok import segmenter as syntok_seg

    start = time.perf_counter()
    for t in texts:
        for paragraph in syntok_seg.process(t):
            for sentence in paragraph:
                for token in sentence:
                    pass  # Force full evaluation
    elapsed = time.perf_counter() - start

    # Get sentences for accuracy check
    sentences = []
    for paragraph in syntok_seg.process(texts[0]):
        for sentence in paragraph:
            sentences.append("".join(t.spacing + t.value for t in sentence).strip())
    return BenchmarkResult("syntok", elapsed, len(texts), sentences)


def benchmark_spacy_sentencizer(texts: list[str]) -> BenchmarkResult:
    """Benchmark spaCy's rule-based sentencizer (no model needed)."""
    try:
        import spacy
        from spacy.lang.en import English
    except ImportError:
        return None

    nlp = English()
    nlp.add_pipe("sentencizer")

    start = time.perf_counter()
    for t in texts:
        doc = nlp(t)
        list(doc.sents)  # Force evaluation
    elapsed = time.perf_counter() - start

    doc = nlp(texts[0])
    sentences = [sent.text for sent in doc.sents]
    return BenchmarkResult("spaCy sentencizer", elapsed, len(texts), sentences)


def benchmark_spacy_senter(texts: list[str]) -> BenchmarkResult:
    """Benchmark spaCy's trained sentence segmenter."""
    try:
        import spacy
    except ImportError:
        return None

    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
    except OSError:
        print("  Warning: en_core_web_sm not installed")
        return None

    start = time.perf_counter()
    for t in texts:
        doc = nlp(t)
        list(doc.sents)
    elapsed = time.perf_counter() - start

    doc = nlp(texts[0])
    sentences = [sent.text for sent in doc.sents]
    return BenchmarkResult("spaCy senter", elapsed, len(texts), sentences)


def benchmark_stanza(texts: list[str]) -> BenchmarkResult:
    """Benchmark Stanford Stanza tokenizer."""
    try:
        import stanza
    except ImportError:
        return None

    try:
        nlp = stanza.Pipeline("en", processors="tokenize", verbose=False)
    except Exception as e:
        print(f"  Warning: stanza failed to initialize: {e}")
        return None

    start = time.perf_counter()
    for t in texts:
        doc = nlp(t)
    elapsed = time.perf_counter() - start

    doc = nlp(texts[0])
    sentences = [sent.text for sent in doc.sentences]
    return BenchmarkResult("stanza", elapsed, len(texts), sentences)


def benchmark_wtpsplit_pytorch(texts: list[str], device: str = "cpu") -> BenchmarkResult:
    try:
        from wtpsplit import SaT
        import torch
    except ImportError:
        return None

    model = SaT("sat-3l-sm")
    if device == "cuda" and torch.cuda.is_available():
        model.half().to("cuda")
        name = "wtpsplit (PyTorch GPU)"
    else:
        name = "wtpsplit (PyTorch CPU)"

    # Warmup
    _ = list(model.split(texts[:10]))
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    result = list(model.split(texts))
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    sentences = list(model.split(texts[0]))
    return BenchmarkResult(name, elapsed, len(texts), sentences)


def benchmark_wtpsplit_ort(texts: list[str], use_gpu: bool = True) -> BenchmarkResult:
    try:
        from wtpsplit import SaT
    except ImportError:
        return None

    providers = ["CPUExecutionProvider"]
    name = "wtpsplit (ORT CPU)"
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        name = "wtpsplit (ORT GPU)"

    try:
        model = SaT("sat-3l-sm", ort_providers=providers)
    except Exception as e:
        print(f"  Warning: {name} failed to initialize: {e}")
        return None

    # Warmup
    _ = list(model.split(texts[:10]))

    start = time.perf_counter()
    result = list(model.split(texts))
    elapsed = time.perf_counter() - start

    sentences = list(model.split(texts[0]))
    return BenchmarkResult(name, elapsed, len(texts), sentences)


def run_speed_benchmark(num_texts: int = 1000, text_type: str = "complex") -> list[BenchmarkResult]:
    """Run speed benchmarks on all available tokenizers."""
    if text_type == "simple":
        texts = [SIMPLE_TEXT] * num_texts
    else:
        texts = [COMPLEX_TEXT] * num_texts

    results = []

    print(f"\nBenchmarking {num_texts} texts ({text_type})...\n")

    # Core tokenizers (always available)
    print("  Running BlingFire...")
    results.append(benchmark_blingfire(texts))

    print("  Running NLTK...")
    results.append(benchmark_nltk(texts))

    print("  Running pySBD...")
    results.append(benchmark_pysbd(texts))

    print("  Running syntok...")
    results.append(benchmark_syntok(texts))

    # spaCy variants
    print("  Running spaCy sentencizer...")
    result = benchmark_spacy_sentencizer(texts)
    if result:
        results.append(result)

    print("  Running spaCy senter...")
    result = benchmark_spacy_senter(texts)
    if result:
        results.append(result)

    # Stanza
    print("  Running stanza...")
    result = benchmark_stanza(texts)
    if result:
        results.append(result)

    # Optional wtpsplit variants
    print("  Running wtpsplit (PyTorch CPU)...")
    result = benchmark_wtpsplit_pytorch(texts, device="cpu")
    if result:
        results.append(result)

    print("  Running wtpsplit (PyTorch GPU)...")
    result = benchmark_wtpsplit_pytorch(texts, device="cuda")
    if result:
        results.append(result)

    print("  Running wtpsplit (ORT CPU)...")
    result = benchmark_wtpsplit_ort(texts, use_gpu=False)
    if result:
        results.append(result)

    print("  Running wtpsplit (ORT GPU)...")
    result = benchmark_wtpsplit_ort(texts, use_gpu=True)
    if result:
        results.append(result)

    return results


def run_accuracy_test() -> None:
    """Test accuracy on edge cases."""
    print("\n" + "=" * 70)
    print("ACCURACY TEST")
    print("=" * 70)

    tokenizers = {
        "BlingFire": lambda t: __import__("blingfire").text_to_sentences(t).split("\n"),
        "NLTK": lambda t: __import__("nltk").sent_tokenize(t),
        "pySBD": lambda t: __import__("pysbd").Segmenter(language="en", clean=False).segment(t),
    }

    # Add spaCy if available
    try:
        import spacy
        from spacy.lang.en import English

        nlp_sentencizer = English()
        nlp_sentencizer.add_pipe("sentencizer")
        tokenizers["spaCy sentencizer"] = lambda t: [s.text for s in nlp_sentencizer(t).sents]

        try:
            nlp_senter = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
            tokenizers["spaCy senter"] = lambda t: [s.text for s in nlp_senter(t).sents]
        except OSError:
            pass
    except ImportError:
        pass

    # Add stanza if available
    try:
        import stanza

        nlp_stanza = stanza.Pipeline("en", processors="tokenize", verbose=False)
        tokenizers["stanza"] = lambda t: [s.text for s in nlp_stanza(t).sentences]
    except ImportError:
        pass

    for case_name, text in EDGE_CASES:
        print(f"\n{case_name}: {text!r}\n")
        for tok_name, tok_fn in tokenizers.items():
            try:
                sentences = tok_fn(text)
                print(f"  {tok_name}: {sentences}")
            except Exception as e:
                print(f"  {tok_name}: ERROR - {e}")


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as a table."""
    # Sort by speed
    results = sorted(results, key=lambda r: r.time_seconds)
    baseline = results[0].time_seconds

    print("\n" + "=" * 70)
    print("SPEED RESULTS")
    print("=" * 70)
    print(f"{'Library':<25} {'Time':<12} {'Per text':<12} {'vs fastest':<12}")
    print("-" * 70)

    for r in results:
        ratio = r.time_seconds / baseline
        print(f"{r.name:<25} {r.time_seconds:.4f}s      {r.ms_per_text:.3f}ms      {ratio:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark sentence tokenizers")
    parser.add_argument(
        "-n", "--num-texts", type=int, default=1000, help="Number of texts to process"
    )
    parser.add_argument("-t", "--text-type", choices=["simple", "complex"], default="complex")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy tests")
    parser.add_argument("--speed", action="store_true", help="Run speed benchmarks")
    args = parser.parse_args()

    # Default to both if neither specified
    if not args.accuracy and not args.speed:
        args.accuracy = True
        args.speed = True

    if args.speed:
        results = run_speed_benchmark(args.num_texts, args.text_type)
        print_results(results)

    if args.accuracy:
        run_accuracy_test()


if __name__ == "__main__":
    main()
