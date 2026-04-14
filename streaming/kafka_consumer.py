"""
kafka_consumer.py
-----------------
Kafka consumer that reads network-traffic records from KAFKA_TOPIC,
runs the ensemble anomaly detector, and publishes alerts to KAFKA_ALERT_TOPIC.

Alert messages are JSON with full DetectionResult payload + original record.

Usage:
    python kafka_consumer.py
"""

import json
import logging
import os
import signal
import sys
import time
from typing import Dict, List

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

from detect_anomaly import DetectionResult, detector

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "network_traffic")
KAFKA_ALERT_TOPIC = os.getenv("KAFKA_ALERT_TOPIC", "intrusion_alerts")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "ids_consumer_group")
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))

BATCH_SIZE = int(os.getenv("CONSUMER_BATCH_SIZE", "32"))
POLL_TIMEOUT = float(os.getenv("CONSUMER_POLL_TIMEOUT", "1.0"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_alert(record: Dict, result: DetectionResult) -> bytes:
    alert = {
        "timestamp": time.time(),
        "is_anomaly": result.is_anomaly,
        "ensemble_label": result.ensemble_label,
        "confidence": result.confidence,
        "model_scores": result.model_scores,
        "model_labels": result.model_labels,
        "original_record": record,
    }
    return json.dumps(alert, default=str).encode("utf-8")


def _delivery_report(err, msg):
    if err:
        logger.error("Alert delivery failed: %s", err)
    else:
        logger.debug(
            "Alert sent to %s [partition %d] offset %d",
            msg.topic(), msg.partition(), msg.offset(),
        )


# ── Consumer loop ─────────────────────────────────────────────────────────────

def run_consumer(
    bootstrap: str = KAFKA_BOOTSTRAP,
    input_topic: str = KAFKA_TOPIC,
    alert_topic: str = KAFKA_ALERT_TOPIC,
    group_id: str = KAFKA_GROUP_ID,
    threshold: float = ANOMALY_THRESHOLD,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Blocking consumer loop.  Sends predictions to *alert_topic* for every record
    and logs anomalies at WARNING level.
    """
    # Pre-load models before entering the hot loop
    logger.info("Loading detection models …")
    detector.load()
    logger.info("Models ready. Starting consumer on topic '%s'.", input_topic)

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": 300_000,
        }
    )

    alert_producer = Producer(
        {
            "bootstrap.servers": bootstrap,
            "linger.ms": 5,
            "compression.type": "lz4",
            "acks": "1",
        }
    )

    consumer.subscribe([input_topic])

    # Graceful shutdown on SIGTERM / SIGINT
    _running = [True]

    def _shutdown(sig, frame):
        logger.info("Shutdown signal received.")
        _running[0] = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    total_processed = 0
    total_anomalies = 0

    pending_records: List[Dict] = []
    pending_messages = []

    try:
        while _running[0]:
            msg = consumer.poll(timeout=POLL_TIMEOUT)

            if msg is None:
                # Flush any partially-filled batch on idle poll
                if pending_records:
                    _flush_batch(
                        pending_records, pending_messages,
                        consumer, alert_producer, alert_topic, threshold,
                        total_processed, total_anomalies,
                    )
                    total_processed += len(pending_records)
                    pending_records.clear()
                    pending_messages.clear()
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug("Reached end of partition %d.", msg.partition())
                else:
                    raise KafkaException(msg.error())
                continue

            try:
                record: Dict = json.loads(msg.value().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning("Skipping malformed message: %s", exc)
                consumer.commit(message=msg, asynchronous=False)
                continue

            pending_records.append(record)
            pending_messages.append(msg)

            if len(pending_records) >= batch_size:
                n_anomalies = _flush_batch(
                    pending_records, pending_messages,
                    consumer, alert_producer, alert_topic, threshold,
                    total_processed, total_anomalies,
                )
                total_processed += len(pending_records)
                total_anomalies += n_anomalies
                pending_records.clear()
                pending_messages.clear()

    finally:
        # Flush stragglers
        if pending_records:
            _flush_batch(
                pending_records, pending_messages,
                consumer, alert_producer, alert_topic, threshold,
                total_processed, total_anomalies,
            )

        alert_producer.flush(timeout=15)
        consumer.close()
        logger.info(
            "Consumer stopped. Processed %d records, %d anomalies detected.",
            total_processed, total_anomalies,
        )


def _flush_batch(
    records, messages,
    consumer, alert_producer, alert_topic, threshold,
    running_total, running_anomalies,
) -> int:
    """Run inference on a batch, publish alerts, commit offsets. Returns anomaly count."""
    results: List[DetectionResult] = detector.predict(records, threshold=threshold)

    n_anomalies = 0
    for record, result in zip(records, results):
        if result.is_anomaly:
            n_anomalies += 1
            logger.warning(
                "ANOMALY detected | confidence=%.3f | scores=%s",
                result.confidence, result.model_scores,
            )

        alert_payload = _make_alert(record, result)
        alert_producer.produce(
            topic=alert_topic,
            value=alert_payload,
            callback=_delivery_report,
        )

    alert_producer.poll(0)
    # Commit offset of the last message in the batch
    consumer.commit(message=messages[-1], asynchronous=False)
    logger.info(
        "Batch of %d processed | %d anomalies | total so far: %d",
        len(records), n_anomalies, running_total + len(records),
    )
    return n_anomalies


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_consumer()