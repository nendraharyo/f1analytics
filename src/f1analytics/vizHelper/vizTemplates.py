import math
from datetime import timedelta

import fastf1
import fastf1.plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import PercentFormatter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# mendefinisikan fungsi


def getDuration(x):  # mendapatkan durasi dari poin A ke B
    min = x.min()
    max = x.max()
    return max - min


def rotate_matrix(
    x, y, angle, x_shift=0, y_shift=0, units="DEGREES"
):  # melakukan proses rotasi matriks numpy
    """
    Rotates a point in the xy-plane counterclockwise through an angle about the origin
    https://en.wikipedia.org/wiki/Rotation_matrix
    :param x: x coordinate
    :param y: y coordinate
    :param x_shift: x-axis shift from origin (0, 0)
    :param y_shift: y-axis shift from origin (0, 0)
    :param angle: The rotation angle in degrees
    :param units: DEGREES (default) or RADIANS
    :return: Tuple of rotated x and y
    """

    # Shift to origin (0,0)
    x = x - x_shift
    y = y - y_shift

    # Convert degrees to radians
    if units == "DEGREES":
        angle = math.radians(angle)

    # Rotation matrix multiplication to get rotated x & y
    xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
    yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

    return xr, yr


def rotate(xy, *, angle):  # melakukan proses rotasi matriks numpy
    rot_mat = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(xy, rot_mat)


def getFastestTelemetry(session):
    fastest = pd.DataFrame()
    for drv in session.drivers:
        try:
            df_temp = pd.DataFrame(
                session.laps.pick_driver(drv)
                .pick_accurate()
                .pick_wo_box()
                .pick_fastest()
                .get_telemetry()
            )
            df_temp["drvName"] = session.get_driver(drv).Abbreviation
            df_temp["teamName"] = session.get_driver(drv).TeamName
            df_temp["teamColor"] = "#" + session.get_driver(drv).TeamColor
            fastest = pd.concat([fastest, df_temp])
        except:
            continue
    return fastest


def getAvgvMinLaptimeClusters(session, clusterNames=None, epsVar=0.44, min_samp=4):
    avg_vs_min_Laptime = (
        session.laps.pick_accurate()
        .pick_wo_box()[["LapTime", "Driver", "DriverNumber"]]
        .groupby(["Driver", "DriverNumber"])
        .agg(bestLap=("LapTime", "min"), avgLap=("LapTime", "mean"))
        .reset_index()
    )
    avg_vs_min_Laptime["color"] = avg_vs_min_Laptime["DriverNumber"].apply(
        lambda x: "#" + session.get_driver(x).TeamColor
    )

    scaler = StandardScaler()
    avg_vs_min_Laptime["bestLapSecs"] = avg_vs_min_Laptime["bestLap"].dt.total_seconds()
    avg_vs_min_Laptime["avgLapSecs"] = avg_vs_min_Laptime["avgLap"].dt.total_seconds()

    train = avg_vs_min_Laptime[["bestLapSecs", "avgLapSecs"]]
    transformed = scaler.fit_transform(train)
    clustering = DBSCAN(eps=epsVar, min_samples=min_samp).fit_predict(transformed)
    avg_vs_min_Laptime["clustering"] = clustering
    avg_vs_min_Laptime["clustering"] = avg_vs_min_Laptime["clustering"].astype(str)
    if clusterNames is not None:
        avg_vs_min_Laptime["Pengelompokan"] = avg_vs_min_Laptime["clustering"].replace(
            avg_vs_min_Laptime["clustering"].unique(), clusterNames
        )
    return avg_vs_min_Laptime


def gettrackSpeedSegmentsDataQuali(laps_telem, session, segments=25):
    laps_telem.loc[:, "miniSect"] = np.round(
        laps_telem["RelativeDistance"].to_numpy() / (1 / segments)
    )

    speed_segments = laps_telem[["Speed", "miniSect", "drvName", "teamColor"]]

    dfSpeed = (
        speed_segments.sort_values("Speed")
        .drop_duplicates(subset="miniSect", keep="last")
        .sort_values("miniSect")
        .reset_index(drop=True)
    )
    listProp = (
        dfSpeed[["drvName", "teamColor"]]
        .value_counts(["drvName", "teamColor"], normalize=True)
        .reset_index()
    )

    single_lap = pd.DataFrame(session.laps.pick_fastest().get_telemetry())[
        ["X", "Y", "RelativeDistance"]
    ]
    single_lap["miniSect"] = single_lap["RelativeDistance"].apply(
        lambda x: int(x / (1 / segments))
    )
    single_lap = single_lap.merge(dfSpeed, on="miniSect")
    single_lap = single_lap.merge(
        single_lap[["drvName"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "id"}),
        on="drvName",
    )  # id dari 0
    single_lap["teamColor"] = single_lap["drvName"].apply(
        lambda x: "#" + session.get_driver(x).TeamColor
    )
    single_lap = single_lap.merge(
        single_lap[["teamColor"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "idTeam"}),
        on="teamColor",
    )  # id dari 0
    return listProp, single_lap


def getAllTelemetry(session, segments=25):
    drvLaps = session.laps[["DriverNumber", "LapNumber"]].groupby("DriverNumber").max()
    all_quali = pd.DataFrame()
    for drv in session.drivers:
        laps = session.laps.pick_driver(drv)

        for i in range(1, int(drvLaps[drvLaps.index == drv]["LapNumber"].iloc[0])):
            try:
                df_temp = pd.DataFrame(
                    laps.pick_lap(i).pick_accurate().pick_wo_box().get_telemetry()
                )
                df_temp["drvName"] = session.get_driver(drv).Abbreviation
                df_temp["teamName"] = session.get_driver(drv).TeamName
                df_temp["teamColor"] = "#" + session.get_driver(drv).TeamColor
                df_temp["LapNumber"] = i
                all_quali = pd.concat([all_quali, df_temp])
            except:
                continue
    all_quali["miniSect"] = (
        np.round(all_quali["RelativeDistance"].to_numpy() / (1 / segments))
    ).astype(int)
    return all_quali


def getFastAgg(all_quali, session):
    first_set = (
        all_quali[["drvName", "Time", "miniSect", "LapNumber", "teamColor", "teamName"]]
        .groupby(["drvName", "miniSect", "LapNumber", "teamColor", "teamName"])
        .agg(getDuration)
        .reset_index()
    )
    theoretical_best_minisectors = (
        first_set[["drvName", "Time", "miniSect", "teamColor", "teamName"]]
        .groupby(["drvName", "miniSect", "teamColor", "teamName"])
        .min()
        .reset_index()
    )
    sum_tbest_minisectors = (
        theoretical_best_minisectors.groupby(["drvName", "teamColor", "teamName"])
        .agg(IdealmSTime=("Time", "sum"))
        .reset_index()
    )

    allLaps = pd.DataFrame(
        session.laps.pick_accurate().pick_wo_box()[
            [
                "Driver",
                "DriverNumber",
                "LapTime",
                "Sector1Time",
                "Sector2Time",
                "Sector3Time",
                "Compound",
                "Team",
            ]
        ]
    )

    fast_agg = (
        allLaps[
            [
                "Driver",
                "DriverNumber",
                "LapTime",
                "Sector1Time",
                "Sector2Time",
                "Sector3Time",
                "Team",
            ]
        ]
        .groupby(["Driver", "DriverNumber", "Team"])
        .agg(
            BestLap=("LapTime", "min"),
            BestS1=("Sector1Time", "min"),
            BestS2=("Sector2Time", "min"),
            BestS3=("Sector3Time", "min"),
        )
    )

    fast_agg["TheoreticalBest"] = (
        fast_agg["BestS1"] + fast_agg["BestS2"] + fast_agg["BestS3"]
    )
    fast_agg.reset_index(inplace=True)
    fast_agg = fast_agg.merge(
        sum_tbest_minisectors, left_on="Driver", right_on="drvName"
    )
    fast_agg.sort_values(by="BestLap", inplace=True)
    return fast_agg


def session_corrected(session):
    laps_corrected = pd.DataFrame()
    laps_num = session.total_laps
    for drv in session.drivers:
        if (laps_num == 0) or (laps_num is None):
            df = (
                (session.laps.pick_driver(drv)["LapNumber"] - 1)
                * (101 / session.laps["LapNumber"].max())
                * 35
            )
        else:
            df = (
                (session.laps.pick_driver(drv)["LapNumber"] - 1) * (101 / laps_num) * 35
            )

        df[:] = df[::-1]
        df = df.apply(lambda x: timedelta(milliseconds=x))

        laps = pd.DataFrame(session.laps.pick_driver(drv))
        laps["fuel_corrected_laptime"] = (laps["LapTime"]) - df

        laps_corrected = pd.concat([laps_corrected, laps])
    return laps_corrected


def getSessionTyreUsage(session_corrected):
    session_tyre = session_corrected[
        [
            "Driver",
            "DriverNumber",
            "LapNumber",
            "Compound",
            "TyreLife",
            "Stint",
            "FreshTyre",
        ]
    ]
    session_tyre = session_tyre.drop(session_tyre[session_tyre["LapNumber"] < 2].index)
    session_tyre = (
        session_tyre.groupby(
            ["Driver", "DriverNumber", "Compound", "Stint", "FreshTyre"]
        )
        .agg({"LapNumber": "count", "TyreLife": "max"})
        .sort_values(by=["Driver", "DriverNumber", "Stint"])
        .reset_index()
    )
    tyregroup = session_tyre[["Driver", "Compound", "TyreLife"]].groupby("Compound")
    tyreAlphas = pd.Series()
    for i, group in tyregroup:
        alphas = group["TyreLife"]
        tyreAlphas = pd.concat(
            [
                tyreAlphas,
                (alphas - (alphas.min() - 2)) / (alphas.max() - (alphas.min() - 2)),
            ]
        )
    session_tyre = session_tyre.merge(
        pd.DataFrame(tyreAlphas), left_index=True, right_index=True
    ).rename(columns={0: "alpha"})
    return session_tyre


def getThrottle(session):
    lap = []
    drivers = []
    throttleDuration = []
    for driver in session.drivers:

        for laps in range(2, session.total_laps):
            drivers.append(session.get_driver(driver).Abbreviation)
            lap.append(laps)
            try:
                test = pd.DataFrame(
                    session.laps.pick_driver(driver).pick_lap(laps).get_telemetry()
                )[["Throttle", "Time"]]
            except:
                throttleDuration.append(None)
                continue

            index = 0
            isFullThrottle = []
            for i, row in test.iterrows():
                if i == 2 and row["Throttle"] >= 98:
                    index += 1
                    isFullThrottle.append(index)
                elif row["Throttle"] >= 98 and test["Throttle"][i - 1] < 98:
                    index += 1
                    isFullThrottle.append(index)
                elif row["Throttle"] >= 98:
                    isFullThrottle.append(index)
                else:
                    isFullThrottle.append(None)
            test["fullThrottleIndex"] = isFullThrottle
            test_group = test.groupby("fullThrottleIndex")
            total_time = timedelta()
            for i, group in test_group:
                total_time += getDuration(group["Time"])
            throttleDuration.append(total_time)
    test_dic = zip(lap, drivers, throttleDuration)

    dfThrottle = pd.DataFrame(list(test_dic)).rename(
        columns={0: "LapNumber", 1: "Driver", 2: "FullThrottleDuration"}
    )

    dfThrottle["tc"] = dfThrottle["Driver"].apply(
        lambda x: "#" + session.get_driver(x).TeamColor
    )

    dfThrottle = dfThrottle.merge(
        session.laps[["LapTime", "LapNumber", "Driver"]], on=["LapNumber", "Driver"]
    ).dropna()
    dfThrottle["PercentFullThrottle"] = (
        dfThrottle["FullThrottleDuration"] / dfThrottle["LapTime"]
    ) * 100
    return dfThrottle


def getPits(laps, drvList):
    laps_adjusted = laps[(laps["Driver"].isin(drvList)) & (laps["LapNumber"] > 1)]
    return laps_adjusted


def findNonGreen(session):
    df = session.laps[["TrackStatus", "LapNumber"]]
    df = df.dropna()
    if (
        len(
            df.drop_duplicates().query(
                '(TrackStatus.str.contains("4")) or (TrackStatus.str.contains("6"))'
            )
        )
        > 0
    ):
        df = (
            df.query(
                '(TrackStatus.str.contains("4")) or (TrackStatus.str.contains("6"))'
            )[["LapNumber", "TrackStatus"]]
            .drop_duplicates()
            .sort_values("LapNumber")
            .reset_index(drop=True)
        )

        df["change"] = df["LapNumber"] - df["LapNumber"].shift(
            1, fill_value=(df["LapNumber"][0]) - 2
        )
        df["group"] = df["LapNumber"].where(df["change"] > 1).ffill()
        return df
    else:
        return None


class vizData:
    def __init__(self, session):

        fastf1.plotting.setup_mpl()
        fastf1.logger.set_log_level("ERROR")

        self.session = session
        print("getting and enriching fastest laps telemetry...")
        self.fastest_laps = getFastestTelemetry(session)
        print("getting and enriching all laps telemetry...")
        self.all_laps = getAllTelemetry(session)
        self.circInfo = session.get_circuit_info()

    def clusterAnalysis(self, clusterNames=None, **kwargs):
        df = getAvgvMinLaptimeClusters(self.session, clusterNames, **kwargs)
        fig, ax = plt.subplots()
        if clusterNames is None:
            hueName = "clustering"
        else:
            hueName = "Pengelompokan"

        sns.scatterplot(data=df, x="avgLap", y="bestLap", hue=hueName, zorder=10, ax=ax)
        texts = [
            ax.annotate(
                row["Driver"],
                [row["avgLap"], row["bestLap"]],
                zorder=10,
                color="lightyellow",
            )
            for i, row in df.iterrows()
        ]

        # ylims = ax.get_ylim()
        # xlims = ax.get_xlim()
        # centre = (max(xlims) / 2, max(ylims) / 2)

        ax.set_xlabel("Rata - Rata Lap")
        ax.set_ylabel("Lap terbaik")
        # props = dict(boxstyle="round", facecolor="w", alpha=0.5)

        adjust_text(texts)
        # colors = (
        #    self.fastest_quali[["teamName", "teamColor"]]
        #    .drop_duplicates()
        #    .set_index("teamName")
        #    .to_dict()["teamColor"]
        # )
        # labels = list(colors.keys())[::-1]
        # handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]

    def trackDominance(self, drvList=None):
        # TODO:start direction rotation
        circRot = self.circInfo.rotation
        lapsTelem = self.all_laps
        if drvList is not None:
            lapsTelem = lapsTelem[lapsTelem["drvName"].isin(drvList)]
        listProp, single_lap = gettrackSpeedSegmentsDataQuali(lapsTelem, self.session)
        x, y = rotate_matrix(single_lap["X"].values, single_lap["Y"].values, circRot)

        listProp.sort_values("proportion", inplace=True)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        fastest_avg = single_lap["idTeam"].to_numpy().astype(float)

        # teamColor=single_lap[['id','tc']].drop_duplicates()
        cmap = ListedColormap(single_lap["teamColor"].drop_duplicates().to_list())
        lc_comp = LineCollection(segments, norm=plt.Normalize(0, cmap.N), cmap=cmap)
        lc_comp.set_array(fastest_avg)
        lc_comp.set_linewidth(4)

        fig, ax = plt.subplots(figsize=(7, 5.5))

        fig.suptitle(f"Kualifikasi {self.session.name} - Track Dominance Semua Laps")

        ax.add_collection(lc_comp)
        ax.axis("equal")
        ax.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

        offset_vector = [700, 0]
        # Iterate over all corners.
        for _, corner in self.circInfo.corners.iterrows():
            # Create a string from corner number and letter
            txt = f"{corner['Number']}{corner['Letter']}"

            # Convert the angle from degrees to radian.
            offset_angle = corner["Angle"] / 180 * np.pi

            # Rotate the offset vector so that it points sideways from the track.
            offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

            # Add the offset to the position of the corner
            text_x = corner["X"] + offset_x
            text_y = corner["Y"] + offset_y

            # Rotate the text position equivalently to the rest of the track map
            text_x, text_y = rotate([text_x, text_y], angle=math.radians(circRot))

            # Rotate the center of the corner equivalently to the rest of the track map
            track_x, track_y = rotate(
                [corner["X"], corner["Y"]], angle=math.radians(circRot)
            )

            # Draw a circle next to the track.
            plt.scatter(text_x, text_y, color="grey", s=140)

            # Draw a line from the track to this circle.
            plt.plot([track_x, text_x], [track_y, text_y], color="grey")

            # Finally, print the corner number inside the circle.
            plt.text(
                text_x,
                text_y,
                txt,
                va="center_baseline",
                ha="center",
                size="small",
                color="white",
            )
        plt.arrow(
            x[0], y[0] + 300, +900, -550, color="white", shape="right", head_width=500
        )
        bounds = [
            i * 10 for i in (listProp.sort_values("proportion")["proportion"].to_list())
        ]
        bounds.insert(0, 0)
        boundsx = []
        mem = 0
        for i in bounds:
            i += mem
            mem = i
            boundsx.append(i)

        cmap = ListedColormap(listProp["teamColor"].to_list())
        norm = BoundaryNorm(boundsx, cmap.N)

        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, spacing="proportional"
        )
        initial = 0
        ticksList = []
        for i in listProp["proportion"]:
            if i != 0:
                ticksList.append((((i * 10) / 2) + initial))
                initial += i * 10

        cbar.set_ticks(ticksList)
        # listProp['bounds']=boundsx
        labels = []
        for j, k in listProp.iterrows():
            label = f'{k["drvName"]} ({round(k["proportion"]*100,2)}%)'
            labels.append(label)

        cbar.set_ticklabels(labels)  # ambil driver name dari variabel!!!
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def sectorComp(self, highlightTeam=None):
        fast_agg = getFastAgg(self.all_laps, self.session)
        fig, ax = plt.subplots(3, 1, figsize=(9, 7))
        fast_agg.sort_values("BestS1", inplace=True)
        for i, row in fast_agg.iterrows():
            if row["Team"] in highlightTeam:
                alphaVar = 1
            else:
                alphaVar = 0.1
            # bottom=np.zeros(len(fast_agg)).astype('<m8[ns]')
            ax[0].bar(
                x=row["Driver"],
                height=row["BestS1"],
                width=0.5,
                color=row["teamColor"],
                alpha=alphaVar,
            )
            # bottom+=row['BestS1'].to_numpy()
            y1, y2 = ax[0].get_ylim()
            y1 = y2 * 0.925
            # y2=y2*0.955
            ax[0].set_ylim(y1, y2)
        fast_agg.sort_values("BestS2", inplace=True)
        for i, row in fast_agg.iterrows():
            if row["Team"] in highlightTeam:
                alphaVar = 1
            else:
                alphaVar = 0.1
            ax[1].bar(
                x=row["Driver"],
                height=row["BestS2"],
                width=0.5,
                color=row["teamColor"],
                alpha=alphaVar,
            )
            # bottom+=row['BestS2'].to_numpy()
            y1, y2 = ax[1].get_ylim()
            y1 = y2 * 0.925
            # y2=y2*0.955
            ax[1].set_ylim(y1, y2)
        fast_agg.sort_values("BestS3", inplace=True)
        for i, row in fast_agg.iterrows():
            if row["Team"] in highlightTeam:
                alphaVar = 1
            else:
                alphaVar = 0.1
            ax[2].bar(
                x=row["Driver"],
                height=row["BestS3"],
                width=0.5,
                color=row["teamColor"],
                alpha=alphaVar,
            )
            y1, y2 = ax[2].get_ylim()
            y1 = y2 * 0.925
            # y2=y2*0.955
            ax[2].set_ylim(y1, y2)
        ax[0].set_title("Sektor Pertama")
        ax[1].set_title("Sektor Kedua")
        ax[2].set_title("Sektor Ketiga")
        # colors = (
        #    fast_agg[["teamName", "teamColor"]]
        #    .drop_duplicates()
        #    .set_index("teamName")
        #    .to_dict()["teamColor"]
        # )
        # labels = list(colors.keys())[::-1]
        # handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
        # fig.legend(handles, labels)
        fig.tight_layout()
        plt.show()

    def exportData(self, path=r"../../data/processed/"):
        with pd.ExcelWriter(
            path + self.session.event.EventName.replace(" ", "") + ".xlsx"
        ) as writer:
            data = vars(self)
            for keys in data.keys():
                if keys not in ["session", "circInfo"]:
                    print(f"Processing {keys}...")
                    data[keys].to_excel(writer, sheet_name=keys, index=False)
            print("Processing Throttle data...")
            getThrottle(self.session).to_excel(
                writer, sheet_name="ThrottleData", index=False
            )


class vizDataQuali(vizData):
    def __init__(self, session):
        vizData.__init__(self, session)

    def quadrantAnalysis(self):
        test = (
            self.fastest_laps.groupby(["drvName", "teamName", "teamColor"])
            .agg(maxSpeed=("Speed", "max"), avgSpeed=("Speed", "mean"))
            .reset_index()
        )
        fig, ax = plt.subplots()

        [
            ax.scatter(
                row["avgSpeed"], row["maxSpeed"], color=row["teamColor"], zorder=10
            )
            for i, row in test.iterrows()
        ]
        texts = [
            ax.annotate(row["drvName"], [row["avgSpeed"], row["maxSpeed"]], zorder=10)
            for i, row in test.iterrows()
        ]
        # ax.axvline(test['avgSpeed'].mean(),color='lightgray')
        # ax.axhline(test['maxSpeed'].mean(),color='lightgray')
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        maxSpeed_center = (test["maxSpeed"].mean() - min(ylims)) / (
            max(ylims) - min(ylims)
        )

        ax.axvspan(
            xmin=test["avgSpeed"].mean(),
            xmax=max(xlims),
            ymin=maxSpeed_center,
            color="g",
            alpha=0.2,
        )
        ax.axvspan(
            xmax=test["avgSpeed"].mean(),
            xmin=min(xlims),
            ymax=maxSpeed_center,
            color="r",
            alpha=0.2,
        )
        ax.axvspan(
            xmax=test["avgSpeed"].mean(),
            xmin=min(xlims),
            ymin=maxSpeed_center,
            color="y",
            alpha=0.2,
        )
        ax.axvspan(
            xmin=test["avgSpeed"].mean(),
            xmax=max(xlims),
            ymax=maxSpeed_center,
            color="b",
            alpha=0.2,
        )

        ax.arrow(
            test["avgSpeed"].mean(),
            min(ylims),
            0,
            max(ylims) - min(ylims),
            color="white",
            head_width=0.2,
        )
        ax.arrow(
            test["avgSpeed"].mean(),
            max(ylims),
            0,
            -(max(ylims) - min(ylims)),
            color="white",
            head_width=0.2,
        )
        ax.arrow(
            min(xlims),
            test["maxSpeed"].mean() - 0.3,
            max(xlims) - min(xlims),
            0,
            color="white",
            head_width=0.2,
        )
        ax.arrow(
            max(xlims),
            test["maxSpeed"].mean() - 0.3,
            -(max(xlims) - min(xlims)),
            0,
            color="white",
            head_width=0.2,
        )

        ax.text(
            max(xlims) + 0.4,
            test["maxSpeed"].mean() + 0.3,
            "cepat",
            rotation=270,
            rotation_mode="anchor",
        )
        ax.text(
            min(xlims) - 0.6,
            test["maxSpeed"].mean() + 0.5,
            "lambat",
            rotation=270,
            rotation_mode="anchor",
        )

        ax.text(test["avgSpeed"].mean() - 0.6, max(ylims) + 0.6, "drag rendah")
        ax.text(test["avgSpeed"].mean() - 0.6, min(ylims) - 0.8, "drag tinggi")
        # props = dict(boxstyle="round", facecolor="w", alpha=0.5)
        ax.text(min(xlims), min(ylims), "kurang efisien", color="r")
        ax.text(max(xlims) - 1, min(ylims) - 0.6, "downforce\ntinggi", color="b")
        ax.text(min(xlims), max(ylims) - 0.4, "downforce\nrendah", color="y")
        ax.text(max(xlims) - 1.8, max(ylims), "efisiensi optimal", color="g")

        ax.set_xlabel("Rata - Rata Kecepatan (Kpj)")
        ax.set_ylabel("Kecepatan Tertinggi (Kpj)")
        adjust_text(texts)
        plt.suptitle("Korelasi Kecepatan Tertinggi dan Rata - Rata Kecepatan")
        plt.title("Diambil dari Lap tercepat masing - masing pembalap", fontsize=9)

        plt.show()


class vizDataRace(vizData):
    def __init__(self, session):
        vizData.__init__(self, session)
        self.session_corrected = session_corrected(session)
        self.tyre_usage = getSessionTyreUsage(self.session_corrected)

    def TyreStrats(self):
        fig, ax = plt.subplots(figsize=(5, 10))
        for driver in self.session.drivers:
            driver_stints = self.tyre_usage.loc[
                self.tyre_usage["DriverNumber"] == driver
            ]

            previous_stint_end = 0
            for idx, row in driver_stints.iterrows():
                # each row contains the compound name and stint length
                # we can use these information to draw horizontal bars

                bars = ax.barh(
                    y=row["Driver"],
                    width=row["LapNumber"],
                    left=previous_stint_end,
                    color=fastf1.plotting.COMPOUND_COLORS[row["Compound"]],
                    edgecolor="black",
                    fill=True,
                    label=row["Compound"],
                    alpha=row["alpha"],
                )

                ax.bar_label(bars, label_type="center", color="black")

                previous_stint_end += row["LapNumber"]

        ax.invert_yaxis()
        tyres = pd.DataFrame(fastf1.plotting.COMPOUND_COLORS, index=[0]).T.reset_index(
            names="Compound"
        )
        tyresLegend = tyres[tyres["Compound"].isin(self.tyre_usage["Compound"])]
        colors = tyresLegend.set_index("Compound").to_dict()[0]
        labels = list(colors.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
        ax.legend(handles, labels)

        ax.set_xlabel("Lap ke -")
        ax.set_ylabel("Pebalap")

        fig.suptitle("         Strategi Ban Sepanjang Balapan", fontsize=15)
        ax.set_title(
            "(warna makin terang menunjukkan penggunaan ban makin panjang)",
            fontsize=6.5,
        )
        fig.tight_layout()
        # fig.legend(driver_stints['Compound'].unique())
        plt.show()

    def TyrePace(self, drvList):
        pits = getPits(self.session_corrected, drvList)
        pits_adjusted = pits[
            (pits["IsAccurate"] == True)
            & (pits["LapTime"] < timedelta(minutes=2, seconds=0))
        ]

        fig, ax = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(12, 8))
        for i, row in pits_adjusted.iterrows():
            if row["Compound"] == "MEDIUM":
                ax[0].scatter(
                    row["LapNumber"],
                    row["fuel_corrected_laptime"],
                    color="#" + self.session.get_driver(row["DriverNumber"]).TeamColor,
                    alpha=0.2,
                )
            if row["Compound"] == "HARD":
                ax[1].scatter(
                    row["LapNumber"],
                    row["fuel_corrected_laptime"],
                    color="#" + self.session.get_driver(row["DriverNumber"]).TeamColor,
                    alpha=0.2,
                )
        ax[0].set_title("MEDIUM", color="yellow", fontsize=10)
        ax[1].set_xlabel("Lap ke-")
        fig.text(-0.01, 0.23, "Waktu per Lap dengan koreksi bahan bakar", rotation=90)
        ax[1].set_title("HARD", fontsize=10)
        stintdf = pits_adjusted[
            [
                "Stint",
                "Compound",
                "fuel_corrected_laptime",
                "LapNumber",
                "Driver",
                "Team",
            ]
        ]
        dfTeamNum = (
            pd.DataFrame(self.session_corrected)[["Driver", "Team", "DriverNumber"]]
            .drop_duplicates()
            .set_index("Driver")
            .groupby("Team")
            .rank()
            .reset_index()
        )
        stintgroups = dfTeamNum.rename(columns={"DriverNumber": "TeamNum"}).merge(
            stintdf, on="Driver"
        )

        drvCache = []
        for i, group in stintgroups.groupby(
            ["Compound", "Driver", "Stint", "Team", "TeamNum"]
        ):
            if i[1] in drvCache:
                labelVar = ""
            else:
                labelVar = i[1]
                drvCache.append(i[1])
            if i[4] == 1:
                linestyleVar = "-"
            else:
                linestyleVar = ":"
            # xx = np.linspace(min(group['fuel_corrected_laptime']),max(group['fuel_corrected_laptime']), 100)
            y = group["fuel_corrected_laptime"].apply(lambda x: x.total_seconds())
            a, b = np.polyfit(group["LapNumber"], y, 1)

            if i[0] == "MEDIUM":
                ax[0].plot(
                    group["LapNumber"],
                    pd.Series(a * group["LapNumber"] + b).apply(
                        lambda x: timedelta(seconds=x)
                    ),
                    color="#" + self.session.get_driver(i[1]).TeamColor,
                    linestyle=linestyleVar,
                    label=labelVar,
                )
            if i[0] == "HARD":
                ax[1].plot(
                    group["LapNumber"],
                    pd.Series(a * group["LapNumber"] + b).apply(
                        lambda x: timedelta(seconds=x)
                    ),
                    color="#" + self.session.get_driver(i[1]).TeamColor,
                    linestyle=linestyleVar,
                    label=labelVar,
                )
        """
        pits_grouped = pits.dropna(subset="PitInTime")[
            ["Driver", "DriverNumber", "LapNumber", "Team"]
        ].groupby("LapNumber")
        for ax in ax:

            for i, group in pits_grouped:

                group.reset_index(inplace=True)
                if len(group["Driver"]) == 2:
                    ax.axvline(
                        x=group["LapNumber"][0],
                        color="#"
                        + self.session.get_driver(group["DriverNumber"][0]).TeamColor,
                        ymax=0.35 / 2,
                    )
                    ax.axvline(
                        x=group["LapNumber"][0],
                        color="#"
                        + self.session.get_driver(group["DriverNumber"][1]).TeamColor,
                        ymin=0.35 / 2,
                        ymax=0.35,
                    )
                    ax.text(
                        group["LapNumber"][0] + 0.17,
                        0.001071,
                        f"{group['Driver'][0]} & {group['Driver'][1]} masuk ke pit",
                        rotation=90,
                        verticalalignment="bottom",
                        fontsize=6.5,
                    )
                else:
                    for i, row in group.iterrows():
                        ax.axvline(
                            x=row["LapNumber"],
                            color="#"
                            + self.session.get_driver(row["DriverNumber"]).TeamColor,
                            ymax=0.35,
                        )
                        ax.text(
                            row["LapNumber"] + 0.17,
                            0.001071,
                            f"{row['Driver']} masuk ke pit",
                            rotation=90,
                            verticalalignment="bottom",
                            fontsize=6.5,
                        )
        """
        fig.suptitle("       Pace per Penggunaan Ban", fontsize=25)

        fig.tight_layout()
        fig.legend()

        plt.show()

    def ThrottleViz(self, drvList):
        pits = getPits(self.session_corrected, drvList).dropna(subset="PitInTime")[
            ["Driver", "DriverNumber", "LapNumber"]
        ]
        dfThrottle = getThrottle(self.session)
        fig, ax = plt.subplots(figsize=(12, 5))
        # sns.lineplot(data=dfThrottle,x='LapNumber',y='FullThrottleDuration',hue='Driver',ax=ax,marker='o')
        tcCache = []

        dfThrottleViz = dfThrottle[
            (dfThrottle["LapNumber"] >= 4) & (dfThrottle["Driver"].isin(drvList))
        ]
        for i, group in dfThrottleViz.groupby(["Driver", "tc"]):
            if i[1] in tcCache:
                linestyleVar = ":"
            else:
                linestyleVar = "-"
            tcCache.append(i[1])
            ax.plot(
                group["LapNumber"],
                group["PercentFullThrottle"],
                color=i[1],
                label=i[0],
                marker="o",
                linestyle=linestyleVar,
            )
            pits_grouped = pits.groupby("LapNumber")

        for i, group in pits_grouped:
            group.reset_index(inplace=True)
            if len(group["Driver"]) == 2:
                ax.axvline(
                    x=group["LapNumber"][0],
                    color="#"
                    + self.session.get_driver(group["DriverNumber"][0]).TeamColor,
                    ymax=0.2 / 2,
                )
                ax.axvline(
                    x=group["LapNumber"][0],
                    color="#"
                    + self.session.get_driver(group["DriverNumber"][1]).TeamColor,
                    ymin=0.2 / 2,
                    ymax=0.2,
                )
                ax.text(
                    group["LapNumber"][0] + 0.45,
                    0,
                    f"{group['Driver'][0]} & {group['Driver'][1]} masuk ke pit",
                    rotation=90,
                    verticalalignment="bottom",
                    fontsize=6.5,
                )
            else:
                for i, row in group.iterrows():
                    ax.axvline(
                        x=row["LapNumber"],
                        color="#"
                        + self.session.get_driver(row["DriverNumber"]).TeamColor,
                        ymax=0.2,
                    )
                    ax.text(
                        row["LapNumber"] + 0.45,
                        0,
                        f"{row['Driver']} masuk ke pit",
                        rotation=90,
                        verticalalignment="bottom",
                        fontsize=6.5,
                    )

        fig.suptitle(
            "Seberapa Lama Para Pembalap Menginjak Pedal Gas secara Penuh per Lap?"
        )
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylabel("prosentase pedal gas penuh per lap")
        ax.set_xlabel("lap ke-")

        ax.legend()

    def BoxPlot(self):
        teamsColor = {
            row["Team"]: "#" + self.session.get_driver(row["Driver"]).TeamColor
            for i, row in pd.DataFrame(
                self.session_corrected[["Team", "Driver"]]
            ).iterrows()
        }

        fig, axs = plt.subplots(figsize=(13, 7))
        df = self.session.laps.pick_wo_box()
        sns.boxplot(
            data=self.session.laps.pick_wo_box(),
            x="Driver",
            y="LapTime",
            hue="Team",
            legend=False,
            palette=teamsColor,
            ax=axs,
            showfliers=False,
            order=df[["Driver", "LapTime"]]
            .groupby("Driver")
            .median()
            .sort_values("LapTime")
            .index,
        )
        lines = axs.lines
        for i in lines:
            i.set_color("w")
        axs.set_title("Konsistensi Pace")
        axs.set_xlabel("pembalap")
        axs.invert_yaxis()

    def LinePlot(self, drvList=None):
        if drvList is None:
            drvList = self.session.drivers
        laps_corrected_lim = self.session_corrected[
            self.session_corrected["LapTime"] < timedelta(minutes=1, seconds=35)
        ]
        fig, axs = plt.subplots(figsize=(13, 7))
        cacheTeam = []
        for index, group in laps_corrected_lim[
            laps_corrected_lim["Driver"].isin(drvList)
        ][["LapTime", "LapNumber", "Driver", "DriverNumber", "Team"]].groupby(
            ["Driver", "DriverNumber", "Team"]
        ):
            if index[2] in cacheTeam:
                linestyleVar = ":"
            else:
                linestyleVar = "-"
                cacheTeam.append(index[2])

            # X_Y_Spline = make_interp_spline(group["LapNumber"], group["LapTime"])
            # X_ = np.linspace(group["LapNumber"].min(), group["LapNumber"].max(), 500)
            # Y_ = X_Y_Spline(X_)
            axs.plot(
                group["LapNumber"],
                group["LapTime"],
                color="#" + self.session.get_driver(index[1]).TeamColor,
                linestyle=linestyleVar,
                label=index[0],
            )
        # sns.lineplot(data=laps_corrected_lim[laps_corrected_lim['LapNumber']>3],y='LapTime',x='LapNumber',hue='Driver',ax=axs[1],palette=teamsColor)
        # axs[1].set_ylim(0.00107,0.00116)
        # fig.suptitle('Konsistensi Pace')
        dfGroupSC = findNonGreen(self.session)
        ymin, ymax = axs.get_ylim()
        if dfGroupSC is not None:
            dfGroupSC = dfGroupSC.groupby("group")

            for i, group in dfGroupSC:
                axs.axvspan(
                    group["LapNumber"].min(),
                    group["LapNumber"].max(),
                    color="yellow",
                    zorder=3,
                )
            # if '4' in group['TrackStatus']:
            # axs.text(group['LapNumber'].min(),((ymax-ymin)/2)+ymin,'SC',c='black',rotation=90,horizontalalignment='left',fontsize=45)
        axs.set_title("Grafik Laptime per Lap")
        axs.set_ylabel("waktu")

        axs.set_ylabel("waktu")
        axs.set_xlabel("lap ke-")

        axs.legend()
        fig.tight_layout()

    def PacePerSector(self, drvList=None):
        if drvList is None:
            drvList = self.session.drivers
        ranks = pd.DataFrame()
        for Lap, group in pd.DataFrame(
            self.session.laps.pick_accurate().pick_wo_box()[
                [
                    "Driver",
                    "LapNumber",
                    "Sector1Time",
                    "Sector2Time",
                    "Sector3Time",
                    "Team",
                ]
            ]
        ).groupby("LapNumber"):
            group["LapNumber"] = Lap
            group.set_index(["LapNumber", "Driver", "Team"], inplace=True)
            ranks = pd.concat([ranks, group.rank(method="dense")])
        ranks = ranks.reset_index()

        fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
        teamCache = []
        for i, group in (
            self.session.laps.pick_accurate()
            .pick_wo_box()[
                [
                    "Driver",
                    "DriverNumber",
                    "LapNumber",
                    "Sector1Time",
                    "Sector2Time",
                    "Sector3Time",
                    "Team",
                ]
            ]
            .groupby(["Driver", "DriverNumber", "Team"])
        ):
            if i[1] not in drvList:
                continue
            if i[2] in teamCache:
                linestyleVar = ":"
            else:
                linestyleVar = "-"
                teamCache.append(i[1])
            axs[0].plot(
                group["LapNumber"],
                group["Sector1Time"],
                marker="o",
                linestyle=linestyleVar,
                color="#" + self.session.get_driver(i[0]).TeamColor,
                label=i[0],
            )

        teamCache = []
        for i, group in (
            self.session.laps.pick_accurate()
            .pick_wo_box()[
                [
                    "Driver",
                    "DriverNumber",
                    "LapNumber",
                    "Sector1Time",
                    "Sector2Time",
                    "Sector3Time",
                    "Team",
                ]
            ]
            .groupby(["Driver", "DriverNumber", "Team"])
        ):
            if i[1] not in drvList:
                continue
            if i[2] in teamCache:
                linestyleVar = ":"
            else:
                linestyleVar = "-"
                teamCache.append(i[1])
            axs[1].plot(
                group["LapNumber"],
                group["Sector2Time"],
                marker="o",
                linestyle=linestyleVar,
                color="#" + self.session.get_driver(i[0]).TeamColor,
                label=i[0],
            )

        teamCache = []
        for i, group in (
            self.session.laps.pick_accurate()
            .pick_wo_box()[
                [
                    "Driver",
                    "DriverNumber",
                    "LapNumber",
                    "Sector1Time",
                    "Sector2Time",
                    "Sector3Time",
                    "Team",
                ]
            ]
            .groupby(["Driver", "DriverNumber", "Team"])
        ):
            if i[1] not in drvList:
                continue
            if i[2] in teamCache:
                linestyleVar = ":"
            else:
                linestyleVar = "-"
                teamCache.append(i[1])
            axs[2].plot(
                group["LapNumber"],
                group["Sector3Time"],
                marker="o",
                linestyle=linestyleVar,
                color="#" + self.session.get_driver(i[0]).TeamColor,
                label=i[0],
            )
        axs[1].set_ylabel("waktu")
        axs[0].set_ylabel("waktu")
        axs[2].set_ylabel("waktu")
        axs[0].set_title("Sektor Pertama")
        axs[1].set_title("Sektor Kedua")
        axs[2].set_title("Sektor Ketiga")
        axs[2].set_xlabel("lap ke-")
        axs[0].legend()
        plt.show()

    def TyreDeg(self, drvList=None):

        pits = getPits(self.session_corrected, drvList)
        pits_adjusted = pits[
            (pits["IsAccurate"] == True)
            & (pits["LapTime"] < timedelta(minutes=2, seconds=0))
        ]
        compound_groups = {}
        tyres = pits_adjusted["Compound"].drop_duplicates()
        fig, ax = plt.subplots(1, len(tyres), sharey=True, sharex=True, figsize=(13, 6))
        for i, group in enumerate(pits_adjusted.groupby("Compound")):
            compound_groups[group[0]] = i
            if group[0] == "SOFT":
                ax[i].set_title("SOFT", color="red", fontsize=10)
            elif group[0] == "MEDIUM":
                ax[i].set_title("MEDIUM", color="yellow", fontsize=10)
            else:
                ax[i].set_title("HARD", fontsize=10)
            for index, row in group[1].iterrows():
                ax[i].scatter(
                    row["TyreLife"],
                    row["fuel_corrected_laptime"],
                    color="#" + self.session.get_driver(row["DriverNumber"]).TeamColor,
                    alpha=0.2,
                )

        fig.text(0.5, -0.01, "Umur Ban")
        fig.text(-0.01, 0.23, "Waktu per Lap dengan koreksi bahan bakar", rotation=90)

        stintdf = pits_adjusted[
            [
                "Stint",
                "Compound",
                "fuel_corrected_laptime",
                "LapNumber",
                "Driver",
                "Team",
                "TyreLife",
            ]
        ]
        dfTeamNum = (
            pd.DataFrame(self.session_corrected)[["Driver", "Team", "DriverNumber"]]
            .drop_duplicates()
            .set_index("Driver")
            .groupby("Team")
            .rank()
            .reset_index()
        )
        stintgroups = dfTeamNum.rename(columns={"DriverNumber": "TeamNum"}).merge(
            stintdf, on="Driver"
        )

        drvCache = []
        for i, group in stintgroups.groupby(
            ["Compound", "Driver", "Stint", "Team", "TeamNum"]
        ):
            if i[1] in drvCache:
                labelVar = ""
            else:
                labelVar = i[1]
                drvCache.append(i[1])
            if i[4] == 1:
                linestyleVar = "-"
            else:
                linestyleVar = ":"
            # xx = np.linspace(min(group['fuel_corrected_laptime']),max(group['fuel_corrected_laptime']), 100)
            y = group["fuel_corrected_laptime"].apply(lambda x: x.total_seconds())
            a, b = np.polyfit(group["TyreLife"], y, 1)

            ax[compound_groups[i[0]]].plot(
                group["TyreLife"],
                pd.Series(a * group["TyreLife"] + b).apply(
                    lambda x: timedelta(seconds=x)
                ),
                color="#" + self.session.get_driver(i[1]).TeamColor,
                linestyle=linestyleVar,
                label=labelVar,
            )

        fig.suptitle("     Degradasi Ban", fontsize=25)

        fig.tight_layout()
        fig.legend()

        plt.show()

    def drsGain(self):
        # TODO: add error correction
        drsSpeed = self.all_laps[
            [
                "drvName",
                "teamName",
                "teamColor",
                "DRS",
                "Speed",
                "DistanceToDriverAhead",
                "Throttle",
                "Brake",
            ]
        ]

        no_drs = drsSpeed[
            (drsSpeed["DRS"].isin([0, 1, 8]))
            & (drsSpeed["Throttle"] > 98)
            & (drsSpeed["Brake"] == 0)
        ]
        no_drs_slipstream = no_drs[no_drs["DistanceToDriverAhead"] < 5]
        no_drs_no_slipstream = no_drs[no_drs["DistanceToDriverAhead"] > 5]

        with_drs = drsSpeed[~drsSpeed["DRS"].isin([0, 1, 2, 3, 8])]
        drs_slipstream = with_drs[with_drs["DistanceToDriverAhead"] < 5]
        drs_no_slipstream = with_drs[with_drs["DistanceToDriverAhead"] > 5]
        # no_drs.drop([646,653,644,643,642,666,641,651,630,637,636],inplace=True)
        maxspeed_no_drs = (
            no_drs[["drvName", "teamName", "teamColor", "Speed"]]
            .groupby(["drvName", "teamName", "teamColor"])
            .max()
        )
        maxspeed_drs = (
            with_drs[["drvName", "teamName", "teamColor", "Speed"]]
            .groupby(["drvName", "teamName", "teamColor"])
            .max()
        )

        result = maxspeed_drs - maxspeed_no_drs
        result.rename(columns=({"Speed": "SpeedGain"}), inplace=True)
        result["SpeedGainStr"] = [
            "" if math.isnan(x) == True else ("+" + str(x)) for x in result["SpeedGain"]
        ]
        maxspeed_drs.merge(result, on="drvName")

        teamsColor = {
            row["teamName"]: row["teamColor"]
            for i, row in drsSpeed[["teamName", "teamColor"]]
            .drop_duplicates()
            .iterrows()
        }
        maxspeed_no_drs.sort_values("Speed", inplace=True, ascending=False)
        fig, ax = plt.subplots(figsize=(12, 5))

        sns.barplot(
            data=maxspeed_drs,
            x="drvName",
            y="Speed",
            color="orange",
            ax=ax,
            label="Dengan DRS",
            order=result.sort_values("SpeedGain", ascending=False).reset_index()[
                "drvName"
            ],
            width=0.6,
        )

        sns.barplot(
            data=maxspeed_no_drs,
            x="drvName",
            y="Speed",
            ax=ax,
            color="green",
            label="Tanpa DRS",
            order=result.sort_values("SpeedGain", ascending=False).reset_index()[
                "drvName"
            ],
            width=0.6,
        )
        ax.bar_label(
            ax.containers[1],
            labels=result.sort_values("SpeedGain", ascending=False)["SpeedGainStr"],
            fontsize=6,
            # rotation=90,
            padding=4,
            color="black",
        )
        ax.set_xlabel("Pembalap")
        ax.set_ylabel("Kecepatan (kpj)")

        ax.set_ylim(250, 365)
        fig.suptitle("Kecepatan yang didapatkan saat menggunakan DRS")
        plt.show()
