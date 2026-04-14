"""
kafka_producer.py
-----------------
Kafka producer that streams network-traffic records to the configured topic.

Supports two modes:
  - CSV replay  : reads a UNSW-NB15 CSV and sends rows one-by-one (or in batches)
  - Stdin mode  : reads JSON lines from stdin (useful for piping live captures)

Usage:
    # Replay CSV
    python kafka_producer.py --csv UNSW_NB15_training-set.csv --delay 0.01

    # Pipe JSON lines
    cat live_traffic.jsonl | python kafka_producer.py --stdin
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Iterator, Optional

import pandas as pd
from confluent_kafka import Producer, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration (override via env vars) ─────────────────────────────────────

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "network_traffic")
KAFKA_PARTITIONS = int(os.getenv("KAFKA_PARTITIONS", "3"))
KAFKA_REPLICATION = int(os.getenv("KAFKA_REPLICATION", "1"))

DROP_COLS = ["id", "attack_cat", "label"]


# ── Topic auto-creation ────────────────────────────────────────────────────────

def ensure_topic(bootstrap: str, topic: str, partitions: int, replication: int) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    existing = admin.list_topics(timeout=10).topics
    if topic not in existing:
        new_topic = NewTopic(topic, num_partitions=partitions,
                             replication_factor=replication)
        futures = admin.create_topics([new_topic])
        for t, fut in futures.items():
            try:
                fut.result()
                logger.info("Created Kafka topic: %s", t)
            except Exception as exc:
                logger.warning("Topic creation warning for %s: %s", t, exc)
    else:
        logger.info("Kafka topic '%s' already exists.", topic)


# ── Record generators ─────────────────────────────────────────────────────────

def csv_records(csv_path: str) -> Iterator[Dict]:
    """Yield dicts from a UNSW-NB15 CSV, one row at a time."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    for _, row in df.iterrows():
        yield row.to_dict()


def stdin_records() -> Iterator[Dict]:
    """Yield JSON dicts from stdin (one JSON object per line)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed JSON line: %s", exc)


# ── Delivery callback ─────────────────────────────────────────────────────────

def _delivery_report(err, msg):
    if err:
        logger.error("Message delivery failed: %s", err)
    else:
        logger.debug(
            "Delivered to %s [partition %d] offset %d",
            msg.topic(), msg.partition(), msg.offset(),
        )


# ── Producer ──────────────────────────────────────────────────────────────────

def run_producer(
    records: Iterator[Dict],
    bootstrap: str = KAFKA_BOOTSTRAP,
    topic: str = KAFKA_TOPIC,
    delay: float = 0.0,
    batch_size: int = 1,
) -> None:
    """
    Push *records* to *topic* on the Kafka cluster at *bootstrap*.

    Parameters
    ----------
    delay      : seconds to wait between individual messages (rate-limiting)
    batch_size : flush every N messages
    """
    ensure_topic(bootstrap, topic, KAFKA_PARTITIONS, KAFKA_REPLICATION)

    producer = Producer(
        {
            "bootstrap.servers": bootstrap,
            "linger.ms": 5,
            "compression.type": "lz4",
            "acks": "1",
        }
    )

    sent = 0
    try:
        for record in records:
            payload = json.dumps(record, default=str).encode("utf-8")
            producer.produce(
                topic=topic,
                value=payload,
                callback=_delivery_report,
            )
            sent += 1

            if sent % batch_size == 0:
                producer.poll(0)

            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        logger.info("Interrupted by user after %d messages.", sent)
    except KafkaException as exc:
        logger.error("Kafka error: %s", exc)
        raise
    finally:
        logger.info("Flushing remaining messages …")
        producer.flush(timeout=30)
        logger.info("Producer done. Sent %d records to topic '%s'.", sent, topic)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream network records to Kafka")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Path to UNSW-NB15 CSV file")
    src.add_argument("--stdin", action="store_true", help="Read JSON lines from stdin")

    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds between messages (default: 0)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Flush every N messages (default: 100)")
    parser.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    parser.add_argument("--topic", default=KAFKA_TOPIC)
    args = parser.parse_args()

    record_iter = csv_records(args.csv) if args.csv else stdin_records()
    run_producer(
        records=record_iter,
        bootstrap=args.bootstrap,
        topic=args.topic,
        delay=args.delay,
        batch_size=args.batch_size,
    )