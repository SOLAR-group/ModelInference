import glob
import numpy as np
import os
import pandas as pd

from pandas import DataFrame

path = os.path.join("data")

programs = ["BIRT", "kate", "Vibe", "Calendar", "krita", "LibreOffice_Writer", "Firefox_for_Android", "Firefox_OS",
            "SeaMonkey", "Thunderbird"]

data_frame = DataFrame(columns=["program", "time"], index=None)

for program in programs:
    print(f"{program}")
    time_files = glob.glob(os.path.join(path, program, "*/*/*time*"))
    for time_file in time_files:
        print(f"{time_file}")
        with open(time_file, "r") as file:
            lines = file.read().splitlines()
            data_frame = data_frame.append({"program": program, "time": lines[0]}, ignore_index=True)

data_frame = data_frame.astype({"program": str, "time": float})
data_frame = data_frame.groupby("program").agg({"time": "mean"})
data_frame["time"] = pd.to_datetime(data_frame['time'], unit='s').dt.strftime('%Hh %Mm %Ss')
print(data_frame)
