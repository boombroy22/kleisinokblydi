"""Tile a list of images into a labelled grid: python grid.py out.png cols width img1 img2 ..."""
import os
import sys

from PIL import Image, ImageDraw

out, cols, width = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
paths = sys.argv[4:]
ims = [Image.open(p).convert("RGB") for p in paths]
w = width
h = int(ims[0].size[1] * w / ims[0].size[0])
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w, rows * h), "white")
dr = ImageDraw.Draw(sheet)
for i, (im, p) in enumerate(zip(ims, paths)):
    x, y = (i % cols) * w, (i // cols) * h
    sheet.paste(im.resize((w, h), Image.LANCZOS), (x, y))
    dr.rectangle([x, y, x + 70, y + 16], fill="black")
    dr.text((x + 3, y + 2), os.path.splitext(os.path.basename(p))[0], fill="yellow")
sheet.save(out)
print(out, sheet.size)
