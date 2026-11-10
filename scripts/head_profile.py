"""Radius-vs-height profile of the R6 head (group R66) to keep the collar clear of it."""
import math
import sys

path = sys.argv[1]
cur = None
pts = []
for line in open(path):
    p = line.split()
    if not p:
        continue
    if p[0] == "g":
        cur = p[1]
    elif p[0] == "v" and cur == "R66":
        x, y, z = map(float, p[1:4])
        pts.append((x + 9.0, y - 8.0, z + 8.0))  # torso-centred Roblox coords

bins = {}
for x, y, z in pts:
    key = round(y, 2)
    r = math.hypot(x, z)
    bins[key] = max(bins.get(key, 0.0), r)
for key in sorted(bins):
    print(f"y={key:+.2f}  max_r={bins[key]:.3f}")
