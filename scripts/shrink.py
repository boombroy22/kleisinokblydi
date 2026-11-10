"""Downscale an image so its width is at most N pixels: python shrink.py in.png out.png 1600"""
import sys

from PIL import Image

src, dst, w = sys.argv[1], sys.argv[2], int(sys.argv[3])
im = Image.open(src).convert("RGB")
if im.size[0] > w:
    im = im.resize((w, int(im.size[1] * w / im.size[0])), Image.LANCZOS)
im.save(dst, optimize=True)
print(dst, im.size)
