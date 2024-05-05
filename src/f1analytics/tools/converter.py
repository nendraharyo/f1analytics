import json

import pandas as pd


def tyresetFix(datalist):
    if datalist != []:
        data = datalist[-1]
        return data
    else:
        return [None, None]


def convert_ntt(text, race_drivers=None):

    tlmy_df_raw = json.loads(text.replace("ntt_f(", "[").replace(");", "]"))

    ver = tlmy_df_raw[0]
    timestamp = tlmy_df_raw[1]
    race_data = tlmy_df_raw[2]
    weather_data = tlmy_df_raw[3]
    telemetry_data = tlmy_df_raw[4]
    if len(tlmy_df_raw) == 6:
        race_drivers = tlmy_df_raw[5]

    telemetry_dict = {
        "index": "driver_num",
        0: "status",
        1: "laps",
        2: "laptime",
        3: "position",
        4: "gap_from_leader",
        5: "interval",
        6: "pits",
        7: "is_best",
        8: "speed",
        9: "gear",
        10: "gear_switches",
        11: "drs",
        12: "lap_pos",
        13: "engine_rpm",
        14: "tyreset",
        15: "speedtraps",
        16: "maxspeedtraps",
        17: "sector1",
        18: "sector2",
        19: "sector3",
    }
    if ver >= 9:
        telemetry_dict[20] = "best_times"

    if ver >= 10:
        telemetry_dict[21] = "sector_segments"

    if ver >= 11:
        telemetry_dict[22] = "throttle"
        telemetry_dict[23] = "brake"

    tlmy_df = (
        pd.DataFrame(dict(zip(race_drivers, telemetry_data)))
        .transpose()
        .reset_index()
        .rename(columns=telemetry_dict)
    )
    tlmy_df["timestamp"] = timestamp
    cols = tlmy_df.columns
    if "tyreset" in cols:
        tyreset_latest = tlmy_df["tyreset"].apply(lambda x: tyresetFix(x))
        tlmy_df[["tyre_type", "tyre_age"]] = pd.DataFrame(
            tyreset_latest.tolist(), index=tlmy_df.index
        )

    if "speedtraps" in cols:

        tlmy_df[["speedtraps1", "speedtraps2", "speedtraps3", "speedtraps4"]] = (
            pd.DataFrame(tlmy_df["speedtraps"].tolist(), index=tlmy_df.index)
        )

        tlmy_df[
            ["maxspeedtraps1", "maxspeedtraps2", "maxspeedtraps3", "maxspeedtraps4"]
        ] = pd.DataFrame(tlmy_df["maxspeedtraps"].tolist(), index=tlmy_df.index)
    if "best_times" in cols:

        tlmy_df[["bts1", "bts2", "bts3", "btlap"]] = pd.DataFrame(
            tlmy_df["best_times"].tolist(), index=tlmy_df.index
        )

    if "sector_segments" in cols:

        tlmy_df[["ss_s1", "ss_s2", "ss_s3"]] = pd.DataFrame(
            tlmy_df["sector_segments"].tolist(), index=tlmy_df.index
        )
    weather_dict = {
        0: "weather_state",
        1: "track_temp",
        2: "air_temp",
        3: "humidity",
        4: "wind_dir",
        5: "wind_speed",
        6: "air_pressure",
    }
    race_dict = {
        0: "flag_status",
        1: "race_type",
        2: "time_remaining_in_s",
        3: "timestamp",
    }

    weather_df = pd.DataFrame(weather_data).transpose().rename(columns=weather_dict)
    weather_df["timestamp"] = timestamp

    race_df = pd.DataFrame(race_data).transpose().rename(columns=race_dict)
    race_df["timestamp"] = timestamp

    return (tlmy_df, weather_df, race_df)
