from kafka import KafkaConsumer
import json
import pandas as pd
from detect_anomaly import detect

consumer = KafkaConsumer(
    'network_traffic',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for message in consumer:

    packet = message.value

    df = pd.DataFrame([packet])

    result = detect(df)

    print("Traffic:", result)