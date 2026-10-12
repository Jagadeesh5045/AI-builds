"""Runner: builds the mixed-modality corpus, prints a worked cross-modal
example, then benchmarks Matryoshka truncation (768/512/256/128 dims)."""

from multimodal_embeddings import (
    ToyMultimodalEmbedder, Retriever, build_corpus, evaluate,
    truncate, storage_bytes, CONCEPTS, FULL_DIMS, TRUNC_LEVELS,
)


def main():
    embedder = ToyMultimodalEmbedder()
    retriever, queries = build_corpus(embedder)
    print(f"corpus: {len(retriever)} items "
          f"(10 concepts x text/code/image/audio), {len(queries)} held-out queries\n")

    # -- worked example: one text query over the mixed corpus ----------------
    name, qvec = queries[0]  # refund-policy
    query_text = next(c["query"] for c in CONCEPTS if c["name"] == name)
    print(f'query (text): "{query_text}"')
    print("top-5 across all modalities @768d:")
    cands = Retriever()
    for doc_id, mod, vec in retriever._items:
        if doc_id != f"{name}#text":
            cands.add(doc_id, mod, vec)
    for score, doc_id, mod in cands.search(qvec, k=5, dims=768):
        mark = "  <-- same concept" if doc_id.startswith(name + "#") else ""
        print(f"  {score:+.3f}  [{mod:5s}] {doc_id}{mark}")

    # -- truncation benchmark ------------------------------------------------
    print("\ncross-modal recall (targets = same-concept code/image/audio), k=5:")
    print(f'{"dims":>5} {"recall@5":>8} {"mrr":>6}  storage(40 vecs)')
    for dims in TRUNC_LEVELS:
        r = evaluate(retriever, queries, k=5, dims=dims)
        kb = storage_bytes(len(retriever), dims) / 1024
        print(f'{dims:>5} {r["recall@5"]:>8.2f} {r["mrr"]:>6.2f}  {kb:6.1f} KiB')
    full = storage_bytes(len(retriever), FULL_DIMS)
    small = storage_bytes(len(retriever), 128)
    print(f"\n768d -> 128d: {full} B -> {small} B = {full / small:.1f}x less storage, "
          "one stored vector serves every level (nested prefixes, no re-encoding).")
    print("\nNote: toy proxy on synthetic features, not the real 740M-param model. "
          "Numbers measure the proxy only.")


if __name__ == "__main__":
    main()
