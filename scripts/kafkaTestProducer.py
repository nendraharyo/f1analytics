import requests as rq
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=["192.168.1.77:9094"])
while True:
    data = (
        rq.get(r"https://f1.tfeed.net/tt.js")
        .text.replace("\n", "")
        .replace("\t", "")
        .replace("\r", "")
        .replace("//", "")
    )
    producer.send("quickstart-events", data.encode())
producer.flush()
