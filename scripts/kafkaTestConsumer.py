from kafka import KafkaConsumer

# from f1analytics.tools import converter

# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer("f1tfeed_raw", bootstrap_servers=["192.168.1.77:9094"])
for message in consumer:
    # message value and key are raw bytes -- decode if necessary!
    # e.g., for unicode: `message.value.decode('utf-8')`
    print(message.value)
