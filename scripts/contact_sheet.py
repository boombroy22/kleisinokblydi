"""Combine preview renders (and optionally the reference) into one comparison sheet.

Usage: python contact_sheet.py out.png img1.png img2.png ... [--ref ref.png]
"""
import sys

from PIL import Image

args = sys.argv[1:]
ref = None
if "--ref" in args:
    i = args.index("--ref")
    ref = args[i + 1]
    args = args[:i] + args[i + 2:]
out = args[0]
imgs = []
for p in args[1:]:
    im = Image.open(p)
    if im.mode == "RGBA":  # transparent renders -> white studio background
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    imgs.append(im.convert("RGB"))
h = max(im.size[1] for im in imgs)
imgs = [im.resize((int(im.size[0] * h / im.size[1]), h)) for im in imgs]
w = sum(im.size[0] for im in imgs)
rows = [imgs]
sheet_h = h
if ref:
    r = Image.open(ref).convert("RGB")
    r = r.resize((w, int(r.size[1] * w / r.size[0])))
    sheet_h += r.size[1]
sheet = Image.new("RGB", (w, sheet_h), "white")
x = 0
for im in imgs:
    sheet.paste(im, (x, 0))
    x += im.size[0]
if ref:
    sheet.paste(r, (0, h))
sheet.save(out)
print("saved", out, sheet.size)
