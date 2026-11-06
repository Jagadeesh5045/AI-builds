# Distributed design note: card ledger on YugabyteDB

Scope marker used throughout: **Demonstrated** means it was observed on the
single-node cluster run for this project (see RESULTS.md). **Design
reasoning** means it follows from how YugabyteDB works but was not, and on
one node cannot be, exercised here.

## Hash sharding and tablets

**Design reasoning, with one demonstrated artefact.** Every YSQL table is
split into tablets: contiguous ranges of the hash space of the first primary
key column. A row's tablet is chosen by hashing that column, so rows spread
roughly evenly across tablets with no hot spot on sequential ids, which is
the property a payments write path wants. Tablets are the unit of
replication, load balancing and automatic splitting as data grows.

**Demonstrated:** the August 2026 transactions partition
(`transactions_2026_08`) consists of 2 tablets on this node, with hash ranges
split at the midpoint of the hash space, both led by the single tserver
(listed with yb-admin in RESULTS.md). The pre-split count of 2 comes from the
cluster setting `ysql_num_shards_per_tserver=2` used for this run. Index
tables are sharded the same way.

## Replication: single node RF 1 versus production RF 3

**Demonstrated:** this cluster runs with replication factor 1. Every tablet
has exactly one copy, on the one tserver. There is no fault tolerance at
all: if this node dies, the data is unavailable and, without a backup, gone.
That is acceptable for a local portfolio run and for nothing else.

**Design reasoning:** a production deployment runs an odd replication
factor, normally RF 3, across at least three tservers placed in separate
fault domains (racks or availability zones). Each tablet then has three
copies organised as a Raft group; a majority (2 of 3) must agree on a write,
and the group keeps serving through the loss of any one node with automatic
leader election. Placement rules (cloud, region, zone per node) are what
stop two replicas of the same tablet landing in the same fault domain. None
of this was demonstrated here: with one node there is no second copy, no
election and no failover to observe.

## Consistency

**Demonstrated:** YSQL answered the reconciliation queries in RESULTS.md
with ordinary single-cluster SQL semantics, and the server log for this run
records that read committed isolation is disabled by default in this build:
the embedded PostgreSQL falls back to snapshot (repeatable read) semantics
unless `yb_enable_read_committed_isolation` is turned on. The reconciliation
pack does not depend on the difference, but an application ported from
PostgreSQL expecting read committed should know about it.

**Design reasoning:** within a Raft group, writes are ordered by the
leader and a read at the leader is strongly consistent; follower reads and
geo-distributed reads trade freshness for latency, and YCQL exposes tunable
consistency levels (for example quorum reads and writes) that YSQL does not.
Cross-region active-active behaviour, follower-read staleness bounds and
clock-skew handling (Hybrid Logical Clocks) are documented YugabyteDB
behaviour that a single local node cannot demonstrate.

## When YugabyteDB fits, and when it does not

Choose YugabyteDB over plain PostgreSQL when a workload has PostgreSQL
shaped queries and joins, like this ledger and its reconciliation pack, but
has outgrown one server: write volume or data size beyond a single node, a
hard requirement to survive node or zone loss without an application
rewrite, or a need to place data in more than one region while keeping one
logical database. The card ledger is a fair example of the shape: relational
integrity matters (double entry, foreign keys, check constraints all
enforced here), and the growth axis is transaction volume.

Stay on plain PostgreSQL when one well-sized server carries the load, which
covers most ledgers for a long time. Project 1's PostgreSQL answered the
same statement lookup from the same style of index, and a single node is
operationally far simpler: no Raft groups, no tablet management, full
PostgreSQL feature surface with no beta warnings of the kind ANALYZE
produced here.

Choose Cassandra (Project 2) over YugabyteDB when the access patterns are
known in advance, are simple lookups by key, and do not need joins,
multi-row transactions or ad hoc reconciliation queries; Cassandra's
query-first tables were faster to reason about per pattern and avoid the
cost of a distributed SQL layer, at the price of denormalising per query
and giving up exactly the relational checks this ledger relies on.
YugabyteDB also offers the YCQL API (verified responding on port 9042 in
this run) for workloads that want a Cassandra-like model in the same
cluster, but a design should pick per table, not mix by habit.

## What this project actually proves

A single-node YugabyteDB 2024.2.3.2 cluster ran the Project 1 relational
design essentially unchanged, enforced its constraints, balanced a
100,000-entry double-entry ledger against 50,000 transactions, and answered
the headline queries at the timings in RESULTS.md. Everything about
multi-node replication, failover and geo-distribution above is design
reasoning for a production version, labelled as such, and is not claimed as
tested behaviour.
