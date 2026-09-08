"""Antialiased Unicode labels for BGR UI panels, rendered at display size."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=64)
def ui_font(size, bold=False):
    names = (["C:/Windows/Fonts/malgunbd.ttf"] if bold else []) + [
        "C:/Windows/Fonts/malgun.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def draw_ui_text(image, value, x, baseline, width, size=17,
                 color=(235, 237, 240), bold=False):
    """Fit in one line, with a readable minimum size and ellipsis for overflow."""
    value = str(value)
    font = ui_font(size, bold)
    while font.getlength(value) > width and size > 14:
        size -= 1
        font = ui_font(size, bold)
    if font.getlength(value) > width:
        while value and font.getlength(value + '…') > width:
            value = value[:-1]
        value += '…'
    bounds = font.getbbox(value, anchor='ls')
    left, top = max(0, x + bounds[0]), max(0, baseline + bounds[1])
    right = min(image.shape[1], x + bounds[2] + 2)
    bottom = min(image.shape[0], baseline + bounds[3] + 2)
    if right <= left or bottom <= top:
        return
    tile = Image.fromarray(image[top:bottom, left:right, ::-1].copy())
    ImageDraw.Draw(tile).text((x-left, baseline-top), value, font=font,
                             fill=tuple(reversed(color)), anchor='ls')
    image[top:bottom, left:right] = np.asarray(tile)[:, :, ::-1]
