#!/usr/bin/env python3
"""Generate a printable sheet of AprilTag (tag36h11) markers for the cube-stacking task.

US Letter paper, 300 DPI.  Print at 100% actual size — do NOT use "fit to page".

Usage:
    python scripts/generate_cube_tags.py

Output:
    mile_franka/assets/cube_tags_letter.png

Prints the tag_size value to pass to your AprilTag pose estimator.
"""
import pathlib

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:
    raise SystemExit("opencv-python required: pip install opencv-python-headless")

# ── geometry constants ────────────────────────────────────────────────────────
DPI        = 300
IN_TO_MM   = 25.4

PAPER_W_IN = 8.5
PAPER_H_IN = 11.0
PADDING_MM = 10.0   # margin on all four sides
GUTTER_MM  = 2.0    # gap between adjacent tag cells (for cutting guides)
LABEL_MM   = 5.0    # text strip below each marker image

CUBE_MM    = 50.0   # 5 cm cube face — one tag occupies one cube face
TAG_IDS    = [0, 1] # 0 = bottom_cube, 1 = top_cube
LABELS     = {0: "ID 0  bottom", 1: "ID 1  top"}

GUIDE_GRAY = 200    # cutting-guide line colour (0=black, 255=white)


def mm2px(mm: float) -> int:
    return round(mm / IN_TO_MM * DPI)


def in2px(inches: float) -> int:
    return round(inches * DPI)


# ── page and grid geometry ────────────────────────────────────────────────────
page_w = in2px(PAPER_W_IN)   # 2550 px
page_h = in2px(PAPER_H_IN)   # 3300 px

cell   = mm2px(CUBE_MM)       # 591 px ≈ 50 mm  (one cube face)
gutter = mm2px(GUTTER_MM)     #  24 px ≈  2 mm
pad    = mm2px(PADDING_MM)    # 118 px ≈ 10 mm
lblh   = mm2px(LABEL_MM)      #  59 px ≈  5 mm

usable_w = page_w - 2 * pad
usable_h = page_h - 2 * pad

n_cols = (usable_w + gutter) // (cell + gutter)  # 3
n_rows = (usable_h + gutter) // (cell + gutter)  # 5

grid_w = n_cols * cell + (n_cols - 1) * gutter
grid_h = n_rows * cell + (n_rows - 1) * gutter

# centre the grid on the page
ox = (page_w - grid_w) // 2
oy = (page_h - grid_h) // 2


# ── single-cell renderer ──────────────────────────────────────────────────────
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)


def make_cell(tag_id: int) -> Image.Image:
    marker_side = cell - lblh               # square marker image (below: label strip)
    m_x         = (cell - marker_side) // 2  # horizontal centering offset

    raw      = cv2.aruco.drawMarker(aruco_dict, tag_id, marker_side)
    cell_img = Image.new("L", (cell, cell), 255)
    cell_img.paste(Image.fromarray(raw), (m_x, 0))

    draw = ImageDraw.Draw(cell_img)
    text = LABELS.get(tag_id, f"ID {tag_id}")
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                  size=mm2px(3))
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        font = None
        tw, th = draw.textsize(text)  # fallback: default bitmap font
    tx = (cell - tw) // 2
    ty = marker_side + (lblh - th) // 2
    draw.text((tx, ty), text, fill=0, font=font)
    return cell_img


# ── assemble the sheet ────────────────────────────────────────────────────────
sheet = Image.new("L", (page_w, page_h), 255)
draw  = ImageDraw.Draw(sheet)

for i in range(n_cols * n_rows):
    row, col = divmod(i, n_cols)
    tag_id   = TAG_IDS[i % len(TAG_IDS)]
    x = ox + col * (cell + gutter)
    y = oy + row * (cell + gutter)
    sheet.paste(make_cell(tag_id), (x, y))


# ── cutting guides: full-page light-gray lines in the centre of each gutter ──
def col_guide_x(col: int) -> int:
    if col == 0:        return ox
    if col == n_cols:   return ox + grid_w
    return ox + col * (cell + gutter) - gutter // 2


def row_guide_y(row: int) -> int:
    if row == 0:        return oy
    if row == n_rows:   return oy + grid_h
    return oy + row * (cell + gutter) - gutter // 2


for col in range(n_cols + 1):
    xg = col_guide_x(col)
    draw.line([(xg, 0), (xg, page_h)], fill=GUIDE_GRAY, width=1)

for row in range(n_rows + 1):
    yg = row_guide_y(row)
    draw.line([(0, yg), (page_w, yg)], fill=GUIDE_GRAY, width=1)


# ── save ──────────────────────────────────────────────────────────────────────
out = (pathlib.Path(__file__).resolve().parent.parent
       / "mile_franka" / "assets" / "cube_tags_letter.png")
sheet.save(str(out), dpi=(DPI, DPI))


# ── report ────────────────────────────────────────────────────────────────────
# cv2.aruco.drawMarker renders the full marker including the 1-cell white quiet
# zone on each side.  For tag36h11 (10 cells total = 2 quiet + 8 data+border):
#   full printed size  = marker_side / DPI * 25.4 / 1000  metres
#   tag_size (apriltag pose estimator) = 8/10 of the above (outer black border)
marker_m   = (cell - lblh) / DPI * IN_TO_MM / 1000
tag_size_m = round(marker_m * 8 / 10, 4)

n0 = sum(1 for i in range(n_cols * n_rows) if TAG_IDS[i % len(TAG_IDS)] == 0)
n1 = n_cols * n_rows - n0

print(f"Saved:  {out}")
print(f"Layout: {n_cols} cols × {n_rows} rows = {n_cols * n_rows} tags  "
      f"({n0} × ID 0 bottom,  {n1} × ID 1 top)")
print(f"tag_size for detector:  {tag_size_m} m  ({tag_size_m * 100:.2f} cm)")
print("Print at 100% actual size — do NOT scale to fit page")
