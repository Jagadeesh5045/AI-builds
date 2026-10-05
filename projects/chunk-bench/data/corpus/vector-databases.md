# Vector Databases

A vector database stores embeddings and serves approximate nearest neighbour search at scale.
Exact nearest neighbour search is too slow once a collection passes a few hundred thousand
vectors, so these systems trade a small amount of recall for large speedups.

## Index algorithms

HNSW, short for Hierarchical Navigable Small World, builds a layered graph where search starts
at a sparse top layer and greedily descends to denser layers. It offers excellent recall at
millisecond latency and is the default index in most vector databases. IVF, or inverted file
index, clusters vectors into Voronoi cells and only searches the cells nearest the query; it
uses less memory than HNSW but needs periodic retraining as data drifts.

## Popular options

Pinecone and Weaviate are managed services that hide index tuning behind an API. ChromaDB and
Qdrant can be self-hosted in a single container, which suits prototypes and regulated data.
pgvector adds vector search to PostgreSQL, letting teams keep vectors beside relational data
without a new database. The right choice usually depends more on operations than on raw
benchmarks.

## Filtering

Production retrieval almost always combines vector similarity with metadata filters, such as
tenant id, document date, or access permissions. Pre-filtering applies the filter before the
vector search and guarantees correct results; post-filtering searches first and discards
non-matching hits, which is faster but can return too few results on selective filters.
