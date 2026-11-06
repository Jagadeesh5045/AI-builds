"""Event transport.

Tries a real Kafka broker first (localhost:9092, topic transaction-recorded).
If no broker is reachable, falls back to a durable local append-only JSON
lines event log. The fallback is a file transport, it is NOT Kafka, and the
pipeline records which transport actually ran.
"""
import json
import os

TOPIC = "transaction-recorded"


def make_event(txn):
    return {
        "event_id": f"evt-{txn[0]}",
        "event_type": "TransactionRecorded",
        "txn_id": txn[0],
        "card_id": txn[1],
        "member_id": txn[2],
        "merchant_id": txn[3],
        "auth_id": txn[4],
        "amount_pence": txn[5],
        "currency": txn[6],
        "txn_type": txn[7],
        "status": txn[8],
        "transaction_date": str(txn[9]),
    }


class KafkaTransport:
    name = "kafka"

    def __init__(self):
        from kafka import KafkaConsumer, KafkaProducer

        self.producer = KafkaProducer(
            bootstrap_servers="127.0.0.1:9092",
            value_serializer=lambda v: json.dumps(v).encode(),
            acks="all",
            request_timeout_ms=10000,
        )
        # a metadata probe: raises if the broker is not really there
        self.producer.partitions_for(TOPIC, timeout=10)
        self.consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers="127.0.0.1:9092",
            value_deserializer=lambda v: json.loads(v.decode()),
            auto_offset_reset="earliest",
            consumer_timeout_ms=15000,
            enable_auto_commit=False,
            group_id=None,
        )

    def publish(self, events):
        for e in events:
            self.producer.send(TOPIC, e)
        self.producer.flush()

    def consume_all(self, expected):
        out = []
        for msg in self.consumer:
            out.append(msg.value)
            if len(out) >= expected:
                break
        self.consumer.close()
        return out


class FileLogTransport:
    """Durable append-only JSON lines event log. Explicit fallback, not Kafka."""

    name = "file_event_log"

    def __init__(self, base_dir):
        self.path = os.path.join(base_dir, "event_log.jsonl")
        if os.path.exists(self.path):
            os.remove(self.path)

    def publish(self, events):
        with open(self.path, "a") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

    def consume_all(self, expected):
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]


def broker_reachable(timeout=2.0):
    import socket

    try:
        with socket.create_connection(("127.0.0.1", 9092), timeout=timeout):
            return True
    except OSError:
        return False


def get_transport(base_dir):
    # Fast probe first: kafka-python metadata retries can block for minutes
    # against a dead broker. If something is listening, use the real client.
    if broker_reachable():
        try:
            return KafkaTransport()
        except Exception:
            pass
    return FileLogTransport(base_dir)
