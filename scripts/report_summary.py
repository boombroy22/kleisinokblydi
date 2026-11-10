"""Print the numbers needed for the README from output/build_report.json (plain python3)."""
import json
import sys

rep = json.load(open(sys.argv[1]))
bmin, bmax = rep["bbox_min"], rep["bbox_max"]
c = [(a + b) / 2 for a, b in zip(bmin, bmax)]
size = [b - a for a, b in zip(bmin, bmax)]
# Blender (x, y, z) -> Roblox torso space (X = -x, Y = z, Z = y)
print("triangles:", rep["final_triangles"], "vertices:", rep["vertices"], "bones:", rep["bones"])
print("max influences:", rep["max_influences"], "unweighted:", rep["unweighted_vertices"],
      "root weighted:", rep["root_weighted_vertices"], "weight sum:", rep["weight_sum_min"], rep["weight_sum_max"])
print("MeshPart size (Roblox X,Y,Z studs): %.3f %.3f %.3f" % (size[0], size[2], size[1]))
print("MeshPart centre relative to Torso (Roblox X,Y,Z): %.3f %.3f %.3f" % (-c[0], c[2], c[1]))
for ch in rep["chains"]:
    print(f"chain {ch['name']:3s} bones={ch['n']} length={ch['ext']}")
print("atlas coverage: %.1f%%" % (100 * rep["atlas_coverage"]))
print("textures:", rep["textures"])
