#!/usr/bin/env python3
"""Benchmark sentence tokenization libraries for speed and accuracy."""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

# Cache directory for downloaded data
CACHE_DIR = Path(__file__).parent / ".cache"

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


@dataclass
class CorpusResult:
    name: str
    corpus: str
    precision: float
    recall: float
    f1: float
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


def calculate_metrics(predicted: list[str], gold: list[str]) -> tuple[float, float, float, int, int, int]:
    """Calculate precision, recall, F1 for sentence boundary detection.

    Compares predicted sentence boundaries against gold standard by checking
    if sentence end positions match.
    """
    # Get end positions of each sentence (cumulative character positions)
    def get_boundaries(sentences: list[str]) -> set[int]:
        boundaries = set()
        pos = 0
        for sent in sentences:
            pos += len(sent)
            boundaries.add(pos)
        return boundaries

    # Normalize: join and get boundaries based on character positions
    pred_text = "".join(predicted)
    gold_text = "".join(gold)

    # If texts don't match, try normalizing whitespace
    if pred_text != gold_text:
        pred_text = " ".join(predicted)
        gold_text = " ".join(gold)

    pred_bounds = get_boundaries(predicted)
    gold_bounds = get_boundaries(gold)

    # Remove final boundary (end of text) - not a real decision point
    if pred_bounds:
        pred_bounds.discard(max(pred_bounds))
    if gold_bounds:
        gold_bounds.discard(max(gold_bounds))

    true_positives = len(pred_bounds & gold_bounds)
    false_positives = len(pred_bounds - gold_bounds)
    false_negatives = len(gold_bounds - pred_bounds)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, true_positives, false_positives, false_negatives


def load_nltk_treebank() -> list[list[str]]:
    """Load NLTK's treebank corpus as list of documents, each a list of sentences."""
    import nltk

    try:
        nltk.data.find("corpora/treebank")
    except LookupError:
        print("  Downloading NLTK treebank corpus...")
        nltk.download("treebank", quiet=True)

    from nltk.corpus import treebank

    documents = []
    for fileid in treebank.fileids():
        sentences = [" ".join(sent) for sent in treebank.sents(fileid)]
        if sentences:
            documents.append(sentences)

    return documents


def load_ud_english() -> list[list[str]]:
    """Load Universal Dependencies English EWT corpus."""
    CACHE_DIR.mkdir(exist_ok=True)
    ud_dir = CACHE_DIR / "ud-english-ewt"

    # Download if not cached
    if not ud_dir.exists():
        print("  Downloading UD English EWT corpus...")
        ud_dir.mkdir(parents=True)

        base_url = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master"
        files = [
            "en_ewt-ud-train.conllu",
            "en_ewt-ud-dev.conllu",
            "en_ewt-ud-test.conllu",
        ]

        for fname in files:
            url = f"{base_url}/{fname}"
            dest = ud_dir / fname
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                print(f"    Warning: Failed to download {fname}: {e}")

    # Parse CoNLL-U files
    documents = []
    for conllu_file in sorted(ud_dir.glob("*.conllu")):
        sentences = []
        current_sentence = []

        with open(conllu_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# text = "):
                    current_sentence = [line[9:]]
                elif line == "" and current_sentence:
                    sentences.extend(current_sentence)
                    current_sentence = []

        if sentences:
            # Group into pseudo-documents of ~20 sentences each
            chunk_size = 20
            for i in range(0, len(sentences), chunk_size):
                chunk = sentences[i:i + chunk_size]
                if chunk:
                    documents.append(chunk)

    return documents


def load_brown() -> list[list[str]]:
    """Load NLTK's Brown corpus as list of documents, each a list of sentences."""
    import nltk

    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        print("  Downloading NLTK Brown corpus...")
        nltk.download("brown", quiet=True)

    from nltk.corpus import brown

    documents = []
    for fileid in brown.fileids():
        sentences = [" ".join(sent) for sent in brown.sents(fileid)]
        if sentences:
            documents.append(sentences)

    return documents


def load_reuters() -> list[list[str]]:
    """Load NLTK's Reuters corpus as list of documents, each a list of sentences."""
    import nltk

    try:
        nltk.data.find("corpora/reuters")
    except LookupError:
        print("  Downloading NLTK Reuters corpus...")
        nltk.download("reuters", quiet=True)

    from nltk.corpus import reuters

    documents = []
    for fileid in reuters.fileids():
        sentences = [" ".join(sent) for sent in reuters.sents(fileid)]
        if sentences:
            documents.append(sentences)

    return documents


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
                for _ in sentence:
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


def benchmark_wtpsplit_pytorch(texts: list[str], device: str = "cpu", model_name: str = "sat-3l-sm") -> BenchmarkResult:
    try:
        from wtpsplit import SaT
        import torch
    except ImportError:
        return None

    model = SaT(model_name)
    if device == "cuda" and torch.cuda.is_available():
        model.half().to("cuda")
        name = f"wtpsplit {model_name} (PyTorch GPU)"
    else:
        name = f"wtpsplit {model_name} (PyTorch CPU)"

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


def benchmark_wtpsplit_ort(texts: list[str], use_gpu: bool = True, model_name: str = "sat-3l-sm") -> BenchmarkResult:
    try:
        from wtpsplit import SaT
    except ImportError:
        return None

    providers = ["CPUExecutionProvider"]
    name = f"wtpsplit {model_name} (ORT CPU)"
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        name = f"wtpsplit {model_name} (ORT GPU)"

    try:
        model = SaT(model_name, ort_providers=providers)
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
    for model_name in ["sat-1l-sm", "sat-3l-sm"]:
        print(f"  Running wtpsplit {model_name} (PyTorch CPU)...")
        result = benchmark_wtpsplit_pytorch(texts, device="cpu", model_name=model_name)
        if result:
            results.append(result)

        print(f"  Running wtpsplit {model_name} (PyTorch GPU)...")
        result = benchmark_wtpsplit_pytorch(texts, device="cuda", model_name=model_name)
        if result:
            results.append(result)

        print(f"  Running wtpsplit {model_name} (ORT CPU)...")
        result = benchmark_wtpsplit_ort(texts, use_gpu=False, model_name=model_name)
        if result:
            results.append(result)

        print(f"  Running wtpsplit {model_name} (ORT GPU)...")
        result = benchmark_wtpsplit_ort(texts, use_gpu=True, model_name=model_name)
        if result:
            results.append(result)

    return results


def get_tokenizers() -> dict[str, callable]:
    """Get all available tokenizers as a dict of name -> function."""
    import blingfire as bf
    import nltk
    import pysbd

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    segmenter = pysbd.Segmenter(language="en", clean=False)

    tokenizers = {
        "BlingFire": lambda t: bf.text_to_sentences(t).split("\n"),
        "NLTK": nltk.sent_tokenize,
        "pySBD": segmenter.segment,
    }

    # syntok
    try:
        from syntok import segmenter as syntok_seg

        def syntok_tokenize(text):
            sentences = []
            for paragraph in syntok_seg.process(text):
                for sentence in paragraph:
                    sentences.append("".join(t.spacing + t.value for t in sentence).strip())
            return sentences

        tokenizers["syntok"] = syntok_tokenize
    except ImportError:
        pass

    # spaCy
    try:
        from spacy.lang.en import English

        nlp_sentencizer = English()
        nlp_sentencizer.add_pipe("sentencizer")
        tokenizers["spaCy sentencizer"] = lambda t: [s.text for s in nlp_sentencizer(t).sents]

        try:
            import spacy
            nlp_senter = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
            tokenizers["spaCy senter"] = lambda t: [s.text for s in nlp_senter(t).sents]
        except (ImportError, OSError):
            pass
    except ImportError:
        pass

    # stanza
    try:
        import stanza
        nlp_stanza = stanza.Pipeline("en", processors="tokenize", verbose=False)
        tokenizers["stanza"] = lambda t: [s.text for s in nlp_stanza(t).sentences]
    except (ImportError, Exception):
        pass

    # wtpsplit
    try:
        from wtpsplit import SaT

        for model_name in ["sat-1l-sm", "sat-3l-sm"]:
            try:
                model = SaT(model_name)
                tokenizers[f"wtpsplit {model_name}"] = lambda t, m=model: list(m.split(t))
            except Exception:
                pass
    except ImportError:
        pass

    return tokenizers


def run_corpus_benchmark(corpora: list[str] | None = None) -> list[CorpusResult]:
    """Run evaluation on real corpora with ground truth."""
    if corpora is None:
        corpora = ["treebank", "ud", "brown", "reuters"]

    results = []

    # Load corpora
    corpus_data = {}
    if "treebank" in corpora:
        print("\nLoading NLTK treebank corpus...")
        corpus_data["treebank"] = load_nltk_treebank()
        print(f"  Loaded {len(corpus_data['treebank'])} documents")

    if "ud" in corpora:
        print("\nLoading UD English EWT corpus...")
        corpus_data["ud"] = load_ud_english()
        print(f"  Loaded {len(corpus_data['ud'])} documents")

    if "brown" in corpora:
        print("\nLoading NLTK Brown corpus...")
        corpus_data["brown"] = load_brown()
        print(f"  Loaded {len(corpus_data['brown'])} documents")

    if "reuters" in corpora:
        print("\nLoading NLTK Reuters corpus...")
        corpus_data["reuters"] = load_reuters()
        print(f"  Loaded {len(corpus_data['reuters'])} documents")

    # Get tokenizers
    print("\nInitializing tokenizers...")
    tokenizers = get_tokenizers()
    print(f"  Found {len(tokenizers)} tokenizers")

    # Evaluate each tokenizer on each corpus
    for corpus_name, documents in corpus_data.items():
        print(f"\nEvaluating on {corpus_name}...")

        for tok_name, tok_fn in tokenizers.items():
            total_tp, total_fp, total_fn = 0, 0, 0

            for gold_sentences in documents:
                # Join sentences to create input text
                text = " ".join(gold_sentences)

                try:
                    predicted = tok_fn(text)
                    _, _, _, tp, fp, fn = calculate_metrics(predicted, gold_sentences)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                except Exception:
                    continue

            # Calculate overall metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results.append(CorpusResult(
                name=tok_name,
                corpus=corpus_name,
                precision=precision,
                recall=recall,
                f1=f1,
                true_positives=total_tp,
                false_positives=total_fp,
                false_negatives=total_fn,
            ))

            print(f"  {tok_name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

    return results


def print_corpus_results(results: list[CorpusResult]) -> None:
    """Print corpus benchmark results as a table."""
    # Group by corpus
    corpora = sorted(set(r.corpus for r in results))

    for corpus in corpora:
        corpus_results = [r for r in results if r.corpus == corpus]
        corpus_results = sorted(corpus_results, key=lambda r: r.f1, reverse=True)

        print(f"\n{'=' * 80}")
        print(f"CORPUS: {corpus.upper()}")
        print("=" * 80)
        print(f"{'Library':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 80)

        for r in corpus_results:
            print(f"{r.name:<25} {r.precision:.4f}       {r.recall:.4f}       {r.f1:.4f}")


def plot_speed_results(results: list[BenchmarkResult], output_path: str = "speed_benchmark.png") -> None:
    """Generate horizontal bar chart for speed benchmark results."""
    import matplotlib.pyplot as plt

    # Sort by speed (fastest first)
    results = sorted(results, key=lambda r: r.ms_per_text)

    names = [r.name for r in results]
    times = [r.ms_per_text for r in results]

    # Color coding: green for fast, yellow for medium, red for slow
    colors = []
    for t in times:
        if t < 0.5:
            colors.append("#2ecc71")  # green
        elif t < 2.0:
            colors.append("#f39c12")  # orange
        else:
            colors.append("#e74c3c")  # red

    fig, ax = plt.subplots(figsize=(10, max(6, len(results) * 0.4)))

    bars = ax.barh(names, times, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, time_val in zip(bars, times):
        ax.text(bar.get_width() + max(times) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{time_val:.3f}ms", va="center", fontsize=9)

    ax.set_xlabel("Time per text (ms)", fontsize=11)
    ax.set_title("Sentence Tokenizer Speed Comparison", fontsize=13, fontweight="bold")
    ax.invert_yaxis()  # Fastest at top
    ax.set_xlim(0, max(times) * 1.15)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSpeed chart saved to: {output_path}")
    plt.close()


def plot_corpus_results(results: list[CorpusResult], output_path: str = "corpus_benchmark.png") -> None:
    """Generate grouped bar chart for corpus evaluation results."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Group by corpus
    corpora = sorted(set(r.corpus for r in results))

    # Get all unique tokenizer names and sort by average F1
    all_names = sorted(set(r.name for r in results))
    avg_f1 = {name: np.mean([r.f1 for r in results if r.name == name]) for name in all_names}
    all_names = sorted(all_names, key=lambda n: avg_f1[n], reverse=True)

    # Build data matrix
    f1_data = {corpus: {} for corpus in corpora}
    for r in results:
        f1_data[r.corpus][r.name] = r.f1

    x = np.arange(len(all_names))
    width = 0.8 / len(corpora)

    fig, ax = plt.subplots(figsize=(12, max(6, len(all_names) * 0.35)))

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    for i, corpus in enumerate(corpora):
        f1_values = [f1_data[corpus].get(name, 0) for name in all_names]
        offset = (i - len(corpora) / 2 + 0.5) * width
        bars = ax.barh(x + offset, f1_values, width * 0.9, label=corpus.upper(),
                       color=colors[i % len(colors)], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title("Sentence Boundary Detection - Corpus Evaluation", fontsize=13, fontweight="bold")
    ax.set_yticks(x)
    ax.set_yticklabels(all_names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right")

    # Add vertical line at F1=0.9 as reference
    ax.axvline(x=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nCorpus chart saved to: {output_path}")
    plt.close()


def run_accuracy_test() -> None:
    """Test accuracy on edge cases."""
    print("\n" + "=" * 70)
    print("ACCURACY TEST (Edge Cases)")
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
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy tests on edge cases")
    parser.add_argument("--speed", action="store_true", help="Run speed benchmarks")
    parser.add_argument("--corpus", action="store_true", help="Run evaluation on real corpora (treebank + UD)")
    parser.add_argument(
        "--corpus-only", choices=["treebank", "ud", "brown", "reuters"], help="Run only specific corpus evaluation"
    )
    parser.add_argument("--plot", action="store_true", help="Generate charts (PNG files)")
    parser.add_argument("--plot-dir", type=str, default=".", help="Directory for chart output")
    args = parser.parse_args()

    # Default to speed + accuracy if nothing specified
    if not args.accuracy and not args.speed and not args.corpus and not args.corpus_only:
        args.accuracy = True
        args.speed = True

    speed_results = None
    corpus_results = None

    if args.speed:
        speed_results = run_speed_benchmark(args.num_texts, args.text_type)
        print_results(speed_results)

    if args.accuracy:
        run_accuracy_test()

    if args.corpus or args.corpus_only:
        corpora = [args.corpus_only] if args.corpus_only else None  # None = all corpora
        corpus_results = run_corpus_benchmark(corpora)
        print_corpus_results(corpus_results)

    # Generate charts if requested
    if args.plot:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        if speed_results:
            plot_speed_results(speed_results, str(plot_dir / "speed_benchmark.png"))

        if corpus_results:
            plot_corpus_results(corpus_results, str(plot_dir / "corpus_benchmark.png"))


if __name__ == "__main__":
    main()
