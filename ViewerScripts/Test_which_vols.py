#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:36:36 2026

@author: jacob.turnbull
"""

"""Plot bvecs"""

import json
import numpy as np
import pandas as pd

# Load JSON
import json

with open("artefacts.json") as f:
    data = json.load(f)

result = {
    "date": [],
    "id": [],
    "vols": []
}

for day in data["day"]:
    date = day["date"]
    for scan in day["scan"]:
        result["date"].append(date)
        result["id"].append(scan["id"])
        result["vols"].append(scan["vols"])

print(result)

# Out put is the following: dictionary with three keys: date, id, vols. Each key has a list of values corresponding to the date, id, and vols from the JSON data.

# Now add all vols together for all dates, i.e concatenate all vols under all entries into a single list or array:

all_vols = []
for vols in result["vols"]:
    all_vols.extend(vols)
print(all_vols)

# Now create a bar chart of all vols using matplotlib: do as barchart and histogram, though bar chart more appropriate for categorical data like vols.

import matplotlib.pyplot as plt

# Name save the above plot as a PNG file:
plt.figure(figsize=(10, 6))
plt.hist(all_vols, bins=np.max(all_vols), edgecolor='black')
plt.xlabel('Vols')
plt.ylabel('Frequency')
plt.title('Histogram of All Vols')
plt.savefig('histogram_of_all_vols.png')

# Count unique subject ID's:
unique_ids = set(result["id"])
print(f"Unique subject IDs: {unique_ids}")
print(f"Number of unique subject IDs: {len(unique_ids)}")
# Sort bins by Frequency; plot with the Volume they correspond to printed on the x axis
plt.close()
import pandas as pd
import matplotlib.pyplot as plt

# count frequencies
counts = pd.Series(all_vols).value_counts().sort_values(ascending=False)

# Use subplots to allow for tight layout and better control over figure size
plt.figure(figsize=(14, 6))
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.2)

# Increase width and height of the plot relative to window
counts.plot(kind='bar')

plt.xlabel("Vols")
plt.ylabel("Frequency")
plt.title("Frequency of Vols (Sorted)")
plt.xticks(rotation=45)
plt.savefig('sorted_histogram_of_all_vols.png')
plt.show()

# Load in the bvecs and bvals for each subject:

import h5py
f = h5py.File('bvec_bval_data.mat','r')
day_1 = f['AR_20250113']
day_2 = f['AR_20250114']
day_3 = f['AR_20250115']
day_4 = f['AR_20250116']
day_5 = f['AR_20250117']
day_6 = f['AR_20250118']
day_7 = f['AR_20250119']

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def unpack_day(day_group):
    
    day_dict = {}
    
    for subj in day_group.keys():
        
        subj_group = day_group[subj]

        if subj.startswith("x"):
            subj_clean = subj[1:]
        else:
            subj_clean = subj
      
        day_dict[subj_clean] = {
            "AP_bvec": np.array(subj_group["AP_bvec"]),
            "PA_bvec": np.array(subj_group["PA_bvec"]),
            "AP_bval": np.array(subj_group["AP_bval"]).flatten(),
            "PA_bval": np.array(subj_group["PA_bval"]).flatten()
        }
        
    return day_dict

day_1_dict = unpack_day(day_1)
day_2_dict = unpack_day(day_2)
day_3_dict = unpack_day(day_3)
day_4_dict = unpack_day(day_4)
day_5_dict = unpack_day(day_5)
day_6_dict = unpack_day(day_6)
day_7_dict = unpack_day(day_7)

# Dictionary with subject keys and corresponding bvecs and bvals data. Each key corresponds to a subject ID, and the value is a numpy array containing the bvecs and bvals data for that subject.

# Now, plot bvecs on a sphere for each subject; use quiver plot to visualize the bvecs as vectors in 3D space. Each vector's direction corresponds to the bvecs, and its length can be scaled by the corresponding bvals.

# Each subject now has a bvals file, bvecs file and a vols file.

# Vols file has the 'bad' volumes for each subject, which we can use to identify which bvecs and bvals correspond to the bad volumes.
# We will then plot this information on a sphere, with the number of 'bad' volumes for a specific bvec represented by the length of the vector in the quiver plot. This will allow us to visually identify which bvecs are associated with a higher number of 'bad' volumes.

# Create a mapping of volume indices to bad volumes for each subject and day


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

# Make subject IDs consistent
bad_vols_by_subject = {}
for idx, subject_id in enumerate(result["id"]):
    subject_id = str(subject_id)
    if subject_id not in bad_vols_by_subject:
        bad_vols_by_subject[subject_id] = result["vols"][idx]

all_subjects = {}
for day_dict in [day_1_dict, day_2_dict, day_3_dict, day_4_dict, day_5_dict, day_6_dict, day_7_dict]:
    for subject_id, subject_data in day_dict.items():
        all_subjects[str(subject_id)] = subject_data

unique_ids = [str(sid) for sid in unique_ids]

# Collect all bad bvec endpoints
points = []

for subject_id in unique_ids:
    if subject_id not in all_subjects or subject_id not in bad_vols_by_subject:
        continue

    bvecs = np.asarray(all_subjects[subject_id]["AP_bvec"])   # (105, 3)
    bad_vols = np.asarray(bad_vols_by_subject[subject_id], dtype=int)

    bad_bvecs = bvecs[bad_vols, :]  # (n_bad, 3)

    # Normalize onto the unit sphere
    norms = np.linalg.norm(bad_bvecs, axis=1)
    bad_bvecs = bad_bvecs[norms > 0]
    norms = norms[norms > 0]
    bad_bvecs = bad_bvecs / norms[:, None]

    points.append(bad_bvecs)

if len(points) == 0:
    raise ValueError("No bad bvecs found to plot.")

points = np.vstack(points)

# Estimate density with 3D bin counts
bins = 30
hist, edges = np.histogramdd(
    points,
    bins=bins,
    range=[[-1, 1], [-1, 1], [-1, 1]]
)

ix = np.clip(np.digitize(points[:, 0], edges[0]) - 1, 0, bins - 1)
iy = np.clip(np.digitize(points[:, 1], edges[1]) - 1, 0, bins - 1)
iz = np.clip(np.digitize(points[:, 2], edges[2]) - 1, 0, bins - 1)

density = hist[ix, iy, iz]

# Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Optional sphere outline
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 20)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(x, y, z, color="gray", alpha=0.15, linewidth=0.5)

sc = ax.scatter(
    points[:, 0], points[:, 1], points[:, 2],
    c=np.log1p(density),
    cmap="viridis",
    s=20,
    alpha=0.9,
    depthshade=False
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Bad bvecs on the unit sphere")

ax.set_box_aspect((1, 1, 1))
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

fig.colorbar(sc, ax=ax, shrink=0.7, label="Local density (log1p count)")
plt.tight_layout()
plt.show()
