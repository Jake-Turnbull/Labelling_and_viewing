#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd

# Load JSON
with open("artefacts.json", "r") as f:
    data = json.load(f)

# Data is stored as a dictionary with three levels: "Date", "Id" and "Vols"

# Convert to Dataframe
records = []
for date, ids in data.items():
    for id_, vols in ids.items():
        records.append({"Date": date, "Id": id_, "Vols": vols})
df = pd.DataFrame(records)

# Take all Vols and flatten them into a single list
all_vols = []
for vols in df["Vols"]:
    all_vols.extend(vols)
# Convert to numpy array for easier analysis
all_vols_array = np.array(all_vols)

# Look at the vols, plot as bar chart:
import matplotlib.pyplot as plt
plt.hist(all_vols_array, bins=np.max(all_vols_array))
plt.xlabel("Vols")
plt.ylabel("Frequency")
plt.title("Distribution of Vols")
plt.show()

# Add the mode to the plot, care which vols are most likely to be affected by the bug
from scipy import stats
mode_vol = stats.mode(all_vols_array)[0][0]
plt.hist(all_vols_array, bins=np.max(all_vols_array))
plt.axvline(mode_vol, color='r', linestyle='dashed', linewidth=1)
plt.xlabel("Vols")
plt.ylabel("Frequency")
plt.title("Distribution of Vols with Mode")
plt.show()
