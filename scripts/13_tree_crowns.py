"""Individual tree crowns from the canopy height model (12_canopy_height.py).

Tree tops are local maxima of the lightly smoothed canopy heights (at least
TOP_MIN_HEIGHT_M tall and TOP_MIN_DISTANCE_M apart); each crown is grown from its
top by a watershed on the inverted heights, within the vegetation pixels only.
Vegetation lower than TOP_MIN_HEIGHT_M with no top nearby (shrubs, hedges) stays in
the raster but gets no crown. The AOI is processed in 1 km tiles with an overlap;
a crown is kept by the tile whose core holds its top, so none is cut or doubled.

Writes outputs/canopy/tree_crowns.gpkg: one polygon per crown with its top
(height_max_m), 95th percentile height, area and an equivalent crown diameter.
Also outputs/canopy/tree_crowns_filled.gpkg: the same crowns with holes smaller than
FILL_HOLES_M2 closed (single non-vegetation pixels inside a crown, 1 m2 each); the
outline is kept, unlike a convex hull, which would swallow the gaps between crowns.

    python 13_tree_crowns.py              # segment crowns, then write both files
    python 13_tree_crowns.py --fill-only  # only remake the filled file from tree_crowns.gpkg
"""

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import shapes
from rasterio.windows import Window
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import config

SRC = config.OUTPUTS / "canopy" / "vegetation_height.tif"
OUT = config.OUTPUTS / "canopy" / "tree_crowns.gpkg"
OUT_FILLED = config.OUTPUTS / "canopy" / "tree_crowns_filled.gpkg"
FILL_HOLES_M2 = 1.02  # a 1 m pixel is a 1 m2 hole; the margin allows for float error
SMOOTH_SIGMA_PX = 1.0
TOP_MIN_HEIGHT_M = 4.0
TOP_MIN_DISTANCE_M = 3
MIN_CROWN_M2 = 4.0
TILE_PX, OVERLAP_PX = 1000, 30


def crowns_in_window(chm, transform, core):
    """Crowns whose top lies in `core` (row0, row1, col0, col1) of this window."""
    veg = chm > 0
    smooth = ndimage.gaussian_filter(np.where(veg, chm, 0), SMOOTH_SIGMA_PX)
    tops = peak_local_max(smooth, min_distance=TOP_MIN_DISTANCE_M, threshold_abs=TOP_MIN_HEIGHT_M,
                          labels=veg.astype(np.uint8), exclude_border=False)
    if len(tops) == 0:
        return []
    markers = np.zeros(chm.shape, np.int32)
    markers[tops[:, 0], tops[:, 1]] = np.arange(1, len(tops) + 1)
    labels = watershed(-smooth, markers, mask=veg)
    r0, r1, c0, c1 = core
    keep = (tops[:, 0] >= r0) & (tops[:, 0] < r1) & (tops[:, 1] >= c0) & (tops[:, 1] < c1)
    keep_ids = set(np.flatnonzero(keep) + 1)
    rows = []
    for geom, v in shapes(labels, mask=labels > 0, transform=transform):
        v = int(v)
        if v in keep_ids:
            rows.append((v, shapely.geometry.shape(geom)))
    if not rows:
        return []
    ids = np.array([v for v, _ in rows])
    # heights per crown from the unsmoothed canopy model
    flat_l, flat_h = labels.ravel(), chm.ravel()
    inside = np.isin(flat_l, ids)
    df = pd.DataFrame({"id": flat_l[inside], "h": flat_h[inside]})
    stats = df.groupby("id")["h"].agg(height_max_m="max", height_p95_m=lambda s: np.percentile(s, 95))
    out = []
    for v, g in rows:
        out.append({"geometry": g, "height_max_m": stats.at[v, "height_max_m"],
                    "height_p95_m": stats.at[v, "height_p95_m"]})
    return out


def fill_small_holes(geom, max_area):
    """The polygon with every interior ring smaller than max_area removed."""
    if geom.geom_type == "MultiPolygon":
        return shapely.MultiPolygon([fill_small_holes(g, max_area) for g in geom.geoms])
    keep = [r for r in geom.interiors if shapely.Polygon(r).area >= max_area]
    return shapely.Polygon(geom.exterior, keep)


def write_filled(src=OUT, out=OUT_FILLED):
    gdf = gpd.read_file(src)
    holes_before = sum(len(g.interiors) for g in gdf.geometry if g.geom_type == "Polygon")
    gdf["geometry"] = [fill_small_holes(g, FILL_HOLES_M2) for g in gdf.geometry]
    holes_after = sum(len(g.interiors) for g in gdf.geometry if g.geom_type == "Polygon")
    gdf["area_m2"] = gdf.area.round(1)
    gdf["crown_diameter_m"] = (2 * np.sqrt(gdf["area_m2"] / np.pi)).round(1)
    out.unlink(missing_ok=True)
    gdf.to_file(out, layer="tree_crowns_filled", driver="GPKG")
    print(f"holes under {FILL_HOLES_M2} m2 closed: {holes_before - holes_after:,} of {holes_before:,} "
          f"({holes_after:,} larger holes kept) -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fill-only", action="store_true", help="only remake tree_crowns_filled.gpkg")
    if parser.parse_args().fill_only:
        write_filled()
        return
    crowns = []
    with rasterio.open(SRC) as src:
        nodata = src.nodata
        for row in range(0, src.height, TILE_PX):
            for col in range(0, src.width, TILE_PX):
                r0, c0 = max(row - OVERLAP_PX, 0), max(col - OVERLAP_PX, 0)
                r1, c1 = min(row + TILE_PX + OVERLAP_PX, src.height), min(col + TILE_PX + OVERLAP_PX, src.width)
                win = Window(c0, r0, c1 - c0, r1 - r0)
                chm = src.read(1, window=win)
                chm = np.where(chm == nodata, 0, chm).astype(np.float32)
                core = (row - r0, row - r0 + TILE_PX, col - c0, col - c0 + TILE_PX)
                crowns += crowns_in_window(chm, src.window_transform(win), core)
            print(f"  rows {row:>5}-{row + TILE_PX}: {len(crowns):,} crowns so far", flush=True)
        crs = src.crs
    gdf = gpd.GeoDataFrame(crowns, crs=crs)
    gdf = gdf[gdf.area >= MIN_CROWN_M2].reset_index(drop=True)
    gdf["area_m2"] = gdf.area.round(1)
    gdf["crown_diameter_m"] = (2 * np.sqrt(gdf["area_m2"] / np.pi)).round(1)
    for col in ("height_max_m", "height_p95_m"):
        gdf[col] = gdf[col].astype(float).round(1)
    OUT.unlink(missing_ok=True)
    gdf.to_file(OUT, layer="tree_crowns", driver="GPKG")
    q = gdf["height_max_m"].quantile([0.1, 0.5, 0.9]).round(1).tolist()
    print(f"{len(gdf):,} crowns; height p10/p50/p90 {q} m; crown diameter median "
          f"{gdf['crown_diameter_m'].median():.1f} m -> {OUT}")
    write_filled()


if __name__ == "__main__":
    main()
