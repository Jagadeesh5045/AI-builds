# RESULTS: Kafka transport retry, 2026-11-06 (evening)

This file records a careful retry of the Kafka run for Project 3, after the
parallel builds that starved the first attempt had finished. The retry failed,
but for a different and now well-evidenced reason. Nothing in this file is
estimated: every claim comes from a command run in this session, and the key
experiments are described so they can be repeated.

## Verdict

Kafka did not run end to end. The broker never reached a serving state in any
attempt, no events were produced or consumed over Kafka (0 produced, 0
consumed), and the pipeline was deliberately NOT rerun in this session, because
run_pipeline.py would silently fall back to the file event log and contaminate
a Kafka results file with fallback numbers. The verified file_event_log run in
RESULTS.md remains the run of record. README wording was left unchanged.

## Environment at retry time

- Free memory (free -m, MemAvailable) moved around during the session: about
  572 MB at the first check, about 854 MB a few minutes later, about 1,277 MB
  at storage format time, about 1,209 MB at the end. Low, but far above the
  roughly 496 MB available during the first attempt, and, as the GC evidence
  below shows, memory was not what killed the broker this time.
- Temurin OpenJDK 17.0.21 was still present at ~/opt-jre and runs.
- Apache Kafka 3.9.1 was still present at ~/opt-kafka.
- No Docker, as before. The two PostgreSQL servers from the earlier projects
  were still running (ports 54329 and the Project 1 socket instance).

## Broker attempts (3 starts, same failure)

Configuration applied before starting: KAFKA_HEAP_OPTS="-Xmx512M -Xms256M"
(heap capped at 512M), log.segment.bytes reduced to 16777216,
offsets.topic.num.partitions set to 1, fresh KRaft storage formatted for each
attempt (log.dirs /tmp/kafka-data2, then /tmp/kafka-data3). The third attempt
also switched controller.quorum.voters and advertised.listeners from localhost
to 127.0.0.1 to rule out name resolution and dual-stack effects.

Every start died within about 2 to 4 seconds, before the broker listener ever
accepted a connection, with the identical error:

    java.lang.OutOfMemoryError: Java heap space
        at java.base/java.nio.HeapByteBuffer.<init>
        at java.base/java.nio.ByteBuffer.allocate
        at org.apache.kafka.common.network.NetworkReceive.readFrom
        ... Selector / NetworkClient.poll ...
        at kafka.server.NodeToControllerRequestThread.doWork

That is the broker-side controller registration channel: in combined KRaft
mode the broker registers with the controller over a real loopback TCP
connection, and something in that read requested a single buffer larger than
the whole heap.

The GC log (~/opt-kafka/logs/kafkaServer-gc.log) proves this was not genuine
heap exhaustion: at exit the G1 heap showed total 524,288K with only 37,999K
used. The heap was almost empty; one allocation request, sized by a length
prefix read from the socket, exceeded it.

## Isolation experiments (the important part)

Three experiments isolate where those bytes come from. All were run in this
session.

1. Java client against nothing. With no broker running (ss showed no listener
   on 9092 or 9093 and no Java process existed), the stock Kafka admin client
   (kafka-broker-api-versions.sh --bootstrap-server 127.0.0.1:9092) still died
   with the same OutOfMemoryError at the same NetworkReceive.readFrom site,
   apparently reading a response. A raw Python socket to the same port at the
   same time received a clean connection refused (ECONNREFUSED), as expected
   with nothing listening.

2. Java client against a controlled listener, loopback. A Python listener was
   bound to 127.0.0.1:19092, set to log any connection and reply with a valid
   zero-length frame. The Kafka admin client was pointed at 127.0.0.1:19092.
   The listener never received any connection at all (its accept timed out),
   yet the Java client still died with the same OutOfMemoryError while
   "reading".

3. Java client against the same listener via the VM interface address. The
   listener was rebound to 0.0.0.0:19092 and the client pointed at the VM's
   interface address (198.19.0.2) port 19092. Same result: the listener never
   received a connection, and the Java client still died with the same error.

Conclusion supported by these experiments: outbound TCP connections made by
Java processes in this sandbox do not reach their destination listener, on
loopback or on the interface address, and reads on those connections return
bytes that are not the peer's, whose first four bytes decode to a length
larger than the heap. Python and C loopback traffic is unaffected (the
PostgreSQL and Cassandra runs earlier today worked over loopback from Python
clients). The exact interception mechanism cannot be identified from inside
this cell; no JAVA_TOOL_OPTIONS, JDK_JAVA_OPTIONS or ld.so.preload was
present in the environment.

Because combined-mode KRaft requires the broker to make exactly such an
outbound Java TCP connection (broker to controller registration) before it can
serve anything, and any Java Kafka client hits the same wall, Kafka cannot
run end to end in this environment as far as these experiments can determine.
Raising the heap would not fix it: the bogus length prefix appears to be of
HTTP-like magnitude (over 1 GB at 512M and at the admin client's default heap
alike), and the connection never reaches a real peer in any case.

## Pipeline and tests in this session

- Events produced to Kafka: 0. Events consumed from Kafka: 0. No topic was
  created (the broker never served a client connection).
- The pipeline itself was not rerun, by design (see Verdict).
- Existing pytest suite, hermetic mode (no CQRS_PG_DSN, DuckDB write model in
  pytest tmp dirs): 3 passed in 0.83s.
- For completeness: the same suite run with CQRS_PG_DSN pointing at the
  persistent cqrs_write database gave 1 failed, 2 passed. The failure is
  explained and is not a transport issue: that database still holds the 20,000
  rows from the verified run in RESULTS.md, so test_totals_match compares the
  full-table write totals against a read model built from the test's 200
  events only.

## State left behind

- No Kafka broker process from this task is running and nothing is listening
  on 9092 or 9093. There was no live broker to stop at the end; every broker
  attempt had already exited on its own.
- ~/opt-kafka/config/kraft/server.properties retains the retry settings
  (log.dirs=/tmp/kafka-data3, log.segment.bytes=16777216,
  offsets.topic.num.partitions=1, 127.0.0.1 in quorum voters and advertised
  listeners), so a future retry outside this sandbox can start from them.
- project3-cqrs/README.md and the capstone README were NOT modified: the
  transport honesty note (Kafka code path exists, verified run used the
  labelled file_event_log) remains accurate.
- Nothing was pushed to GitHub.
