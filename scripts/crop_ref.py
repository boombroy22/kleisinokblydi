"""Crop/zoom regions of the reference turnaround for close study.

Usage: python crop_ref.py <ref.png> <out_dir>
"""
import os
import sys

from PIL import Image

ref, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
im = Image.open(ref).convert("RGB")
print("size", im.size)
w, h = im.size
sx, sy = w / 1536.0, h / 864.0  # crop boxes below are in 1536x864 preview pixels

regions = {
    "front_full": (20, 140, 520, 780),
    "front_shoulders": (30, 200, 510, 420),
    "front_anchor": (190, 240, 360, 420),
    "front_lower": (20, 420, 520, 780),
    "side_full": (600, 140, 980, 780),
    "side_top": (600, 140, 920, 470),
    "back_full": (1010, 140, 1520, 790),
    "back_top": (1010, 170, 1520, 520),
    "back_lantern": (1150, 280, 1350, 520),
    "back_lower": (1010, 480, 1520, 790),
}
for name, (x0, y0, x1, y1) in regions.items():
    box = (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))
    crop = im.crop(box)
    scale = 2 if crop.size[0] < 600 else 1
    crop = crop.resize((crop.size[0] * scale, crop.size[1] * scale), Image.LANCZOS)
    crop.save(os.path.join(out, name + ".png"))
    print(name, crop.size)
