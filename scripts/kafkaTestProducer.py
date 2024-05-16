import json

import requests as rq
from kafka import KafkaProducer

session = rq.session()
with open(r"data/interim/f1tfeed.json") as file:
    cookies = json.load(file)

cookie_props = [
    "name",
    "value",
    "port",
    "domain",
    "path",
    "expires",
    "secure",
    "discard",
]
cookiejar = [
    dict(
        [
            (k, item["session" if k == "discard" else k])
            for k in cookie_props
            if k in item
        ]
    )
    for item in cookies
]

for c in cookiejar:
    session.cookies.set(**c)

producer = KafkaProducer(bootstrap_servers=["192.168.1.77:9094"])
print("Harvesting...")
while True:
    data = (
        session.get(url=r"https://f1.tfeed.net/tt.js")
        .text.replace("\n", "")
        .replace("\t", "")
        .replace("\r", "")
        .replace("//", "")
    )
    if ("ntt_f" in data) & ("ntt_f(11,0,[-1,-1,0],[],[]);" not in data):
        producer.send("f1tfeed_raw", data.encode())
producer.flush()
