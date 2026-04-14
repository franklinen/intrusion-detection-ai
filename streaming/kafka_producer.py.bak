from kafka import KafkaProducer
import pandas as pd
import json
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

df = pd.read_csv("data/UNSW_NB15.csv")

for _, row in df.iterrows():

    producer.send("network_traffic", row.to_dict())

    time.sleep(0.1)