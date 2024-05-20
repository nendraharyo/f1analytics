import os

import pandas as pd
from kafka import KafkaConsumer
from sqlalchemy import create_engine

from f1analytics.tools import converter

engine = create_engine(os.environ.get("stbdb_url"))
conn = engine.connect()

# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer("f1tfeed_raw", bootstrap_servers=["192.168.1.77:9094"])
print("Mulai memproses data..")
for message in consumer:
    # message value and key are raw bytes -- decode if necessary!
    # e.g., for unicode: `message.value.decode('utf-8')`
    # print(message.value)
    try:
        tlmy_df, weather_df, race_df = converter.convert_ntt(message.value.decode())
        if race_df["flag_status"] != "6":
            tlmy_df["timestamp"] = pd.to_datetime(tlmy_df["timestamp"], unit="ms")
            weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], unit="ms")
            race_df["timestamp"] = pd.to_datetime(race_df["timestamp"], unit="ms")
            tlmy_df["throttle"] = tlmy_df["throttle"].astype(int)
            tlmy_df = tlmy_df.set_index("timestamp")
            weather_df_df = weather_df.set_index("timestamp")
            race_df = race_df.set_index("timestamp")
            tlmy_df.to_sql("telemetry_imola_race_2024", conn, if_exists="append")
            weather_df.to_sql("weather_imola_race_2024", conn, if_exists="append")
            race_df.to_sql("racestats_imola_race_2024", conn, if_exists="append")
        else:
            print("sudah ada sinyal checkered flag")
            break
    except:
        print("error")
        continue
