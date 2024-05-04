import json

import requests as rq
from kafka import KafkaProducer

session = rq.session()
with open(r"data\interim\f1tfeed.json") as file:
    cookies = json.load(file)[0]  # save them to file as JSON
    cookies = rq.utils.cookiejar_from_dict(cookies)  # turn dict to cookiejar
    session.cookies.update(cookies)  # load cookiejar to current session


producer = KafkaProducer(bootstrap_servers=["192.168.1.77:9094"])
print("Harvesting...")
while True:
    data = (
        rq.get(r"https://f1.tfeed.net/tt.js")
        .text.replace("\n", "")
        .replace("\t", "")
        .replace("\r", "")
        .replace("//", "")
    )
    if "ntt_f" in data:
        producer.send("quickstart-events", data.encode())
producer.flush()
