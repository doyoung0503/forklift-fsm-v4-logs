"""Continuous planar fork sweep against the measured 3-by-3 pallet blocks.

The forks are constant-width strips extending backwards from their tips. This
conservatively covers any fork length; it does not model heels, vertical fit,
steering drift, pose uncertainty or additional travel beyond the supplied target.
"""
from dataclasses import dataclass
import math

from . import config as cfg


@dataclass(frozen=True)
class InsertionCheck:
    ok: bool
    reason: str
    front_hits: tuple = ()


def geometry_error():
    values = (cfg.INSERT_PALLET_SIZE_M, cfg.INSERT_BLOCK_SIZE_M,
              cfg.INSERT_HOLE_WIDTH_M, cfg.FORK_OUTER_SPAN_M)
    if any(not math.isfinite(v) or v <= 0 for v in values):
        return "invalid pallet/fork dimensions"
    if not math.isclose(3 * cfg.INSERT_BLOCK_SIZE_M + 2 * cfg.INSERT_HOLE_WIDTH_M,
                        cfg.INSERT_PALLET_SIZE_M, abs_tol=1e-9):
        return "pallet dimensions do not sum to the configured size"
    width = cfg.FORK_WIDTH_M
    if width is None:
        return "measured single-fork width is required (FORK_WIDTH_M)"
    if not math.isfinite(width) or not 0 < width < cfg.FORK_OUTER_SPAN_M / 2:
        return "invalid measured single-fork width"
    margin = cfg.INSERT_WALL_MARGIN_M
    if not math.isfinite(margin) or not 0 < margin < cfg.INSERT_HOLE_WIDTH_M / 2:
        return "invalid insertion wall margin"
    return None


def pallet_blocks(margin=0.0):
    """(row, column, polygon), row 5 is the entry face, depth increases inward."""
    size, block = cfg.INSERT_PALLET_SIZE_M, cfg.INSERT_BLOCK_SIZE_M
    pitch = block + cfg.INSERT_HOLE_WIDTH_M
    for iz in range(3):
        for ix in range(3):
            x0, d0 = -size / 2 + ix * pitch, iz * pitch
            yield (5 - 2 * iz, 1 + 2 * ix,
                   ((x0-margin, d0-margin), (x0+block+margin, d0-margin),
                    (x0+block+margin, d0+block+margin), (x0-margin, d0+block+margin)))


def polygons_touch(a, b):
    """Separating-axis test for convex polygons; boundary contact is collision."""
    for polygon in (a, b):
        for i, p in enumerate(polygon):
            q = polygon[(i+1) % len(polygon)]
            nx, ny = -(q[1]-p[1]), q[0]-p[0]
            pa = [nx*x + ny*y for x, y in a]
            pb = [nx*x + ny*y for x, y in b]
            if max(pa) < min(pb) - 1e-12 or max(pb) < min(pa) - 1e-12:
                return False
    return True


def check_insertion_sweep(pose, *, enforce_entry_distance=True):
    """Check entry pockets and every block over the accepted Z-minus-remainder move."""
    error = geometry_error()
    if error:
        return InsertionCheck(False, error)
    try:
        x, z, yaw = float(pose.pallet_x_m), float(pose.pallet_z_m), float(pose.yaw_deg)
    except (AttributeError, TypeError, ValueError):
        return InsertionCheck(False, "missing insertion pose")
    if not all(math.isfinite(v) for v in (x, z, yaw)) or z <= 0:
        return InsertionCheck(False, "invalid insertion pose")
    if enforce_entry_distance and z > cfg.INSERT_ALIGNMENT_MAX_CAMERA_Z_M:
        return InsertionCheck(False, "outside insertion entry distance")
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    if c <= 1e-6:
        return InsertionCheck(False, "pallet face not forward facing")
    target = z - cfg.INSERT_CAMERA_Z_REMAINDER_M
    if target <= 0:
        return InsertionCheck(False, "invalid insertion travel")
    half, width = cfg.FORK_OUTER_SPAN_M / 2, cfg.FORK_WIDTH_M
    centre, tip = cfg.CAMERA_TO_FORK_TIP_X_M, cfg.CAMERA_TO_FORK_TIP_Z_M
    margin = cfg.INSERT_WALL_MARGIN_M
    block, hole = cfg.INSERT_BLOCK_SIZE_M, cfg.INSERT_HOLE_WIDTH_M
    left = -cfg.INSERT_PALLET_SIZE_M / 2 + block
    pockets = ((left, left+hole), (left+hole+block, left+2*hole+block))
    forks = ((centre-half, centre-half+width), (centre+half-width, centre+half))
    hits = ((forks[0][0]-x)/c, (forks[1][1]-x)/c)
    # Map each entire fork into its own front pocket, including inner edges.
    for bounds, pocket in zip(forks, pockets):
        for fx in bounds:
            hit = (fx-x)/c
            if not pocket[0]+margin < hit < pocket[1]-margin:
                return InsertionCheck(False, "front pocket clearance", hits)
            if z-s*(fx-x)/c < tip:
                return InsertionCheck(False, "front face already behind a fork tip", hits)
    # Transform enlarged blocks into the vehicle frame. A forward translation
    # sweeps each strip up to tip+target; extending it backwards avoids assuming
    # an unmeasured fork-root location. SAT checks the continuous sweep, not samples.
    blocks = []
    for row, col, polygon in pallet_blocks(margin):
        blocks.append((row, col, tuple((x+c*u+s*d, z-s*u+c*d) for u, d in polygon)))
    back = min(vz for _, _, polygon in blocks for _, vz in polygon) - 1.0
    # Report nearer pallet rows first, across both forks.
    for row, col, polygon in blocks:
        for side, (lo, hi) in zip(("left", "right"), forks):
            sweep = ((lo, back), (hi, back), (hi, tip+target), (lo, tip+target))
            if polygons_touch(sweep, polygon):
                return InsertionCheck(False, f"{side} fork sweep hits row {row} column {col} (wall margin included)", hits)
    return InsertionCheck(True, "all nine blocks clear over planned insertion", hits)
