#!/usr/bin/env python3
"""Entry point: run the chunking benchmark and write results.

Usage:
    python run_bench.py [--k 5] [--results-dir results]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from chunk_bench.corpus import load_corpus
from chunk_bench.evaluate import format_table, run_benchmark
from chunk_bench.qa import load_qa_pairs

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RAG chunking strategies.")
    parser.add_argument("--k", type=int, default=5, help="Top-k cutoff for metrics.")
    parser.add_argument("--results-dir", default="results", help="Where to write outputs.")
    args = parser.parse_args()

    docs = load_corpus(BASE_DIR / "data" / "corpus")
    pairs = load_qa_pairs(BASE_DIR / "data" / "qa_gold.json")
    print(f"Loaded {len(docs)} documents and {len(pairs)} gold QA pairs.\n")

    df = run_benchmark(docs, pairs, k=args.k)
    table = format_table(df)
    print(f"Chunking benchmark (k={args.k}, {len(pairs)} questions):\n")
    print(table)

    results_dir = BASE_DIR / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "benchmark_results.csv")
    best = df.index[0]
    summary = (
        f"# chunk-bench results\n\n"
        f"Ran {len(pairs)} gold questions over {len(docs)} documents, k={args.k}.\n\n"
        f"Best strategy on recall@{args.k}: **{best}** "
        f"(recall={df.loc[best, 'recall@k']:.3f}, MRR={df.loc[best, 'mrr']:.3f}).\n\n"
        f"Full table:\n\n```\n{table}\n```\n"
    )
    (results_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(f"\nWrote {results_dir / 'benchmark_results.csv'} and {results_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
