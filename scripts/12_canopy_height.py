"""Canopy height model: buildings as vectors, vegetation as a height raster.

From the 1 m nDSM (height above ground), the final building footprints
(outputs/footprints/lidar_classified.gpkg) and their mask (09_object_classifier.py),
and summer Sentinel-2 NDVI (07_sentinel2_ndvi.py):
  buildings   the final footprints, copied with their height columns
  vegetation  nDSM > VEG_MIN_HEIGHT_M, outside every pixel a building footprint touches
              (so the two layers never overlap) and a BUILDING_BUFFER_M ring around the
              building mask (roof edges and eaves would otherwise ring every house),
              after removing structures and noise:
    smooth rule  pixels that are not green (NDVI < STRUCT_NDVI) AND smooth (3x3 height
                 std < STRUCT_ROUGH_M): bridge decks, flat roofs the mask missed,
                 vehicles. Tree canopy is rough (median 2.1 m), bridges are not (0.4 m).
    object rule  connected patches whose mean NDVI < OBJECT_NDVI: masts, cranes,
                 chimneys and pylons, which look rough at their edges but are not green
    noise        anything still above VEG_MAX_HEIGHT_M (99.9% of vegetation is below 33 m)
    mapped bridge  inside a bridge mapped in OpenStreetMap (prepare_osm_structures.py):
                 outlines, and bridge ways (footbridges too) buffered to deck width
NDVI is bilinear-interpolated from 10 m to 1 m, so it changes smoothly across cells.
Buildings and mapped structures (OpenStreetMap bridge decks at their mapped width,
bridge outlines, raised railway platforms):
  - a footprint at least MOSTLY_BRIDGE_SHARE covered by them is a bridge, not a
    building, and is deleted (all such footprints lie outside OS buildings, while
    footprints only touching a structure, e.g. shopping centres with a footbridge,
    are 81% real building and are kept)
  - railway viaducts and bridges, and raised platforms, are cut out of the remaining
    footprints: a viaduct at roof height otherwise merges with the buildings beside it
    (e.g. St Thomas station, Exeter). Pieces left under MIN_BUILDING_M2 and slivers
    narrower than 2 x SLIVER_M are dropped; cut pieces get new heights and keep the
    original id in building_id.

Writes to outputs/canopy/:
  buildings.gpkg          building footprints (vector) with height columns; source is
                          classified (09) or cut_by_rail (viaduct cut)
  vegetation_height.tif   height in m on vegetation pixels, no data elsewhere: click a
                          pixel (QGIS Identify) to read its height
  vegetation_height.qml   QGIS style (green height ramp), loaded with the raster
  removed.tif             tall non-building pixels taken out of vegetation: 1 smooth
                          rule, 2 object rule, 3 above the height cap, 4 mapped bridge
  preview_*.png           sample 500 m windows
and outputs/eval/canopy_summary.csv (overall and per 5 km quadrant).
"""

import importlib
import json

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
from matplotlib.colors import ListedColormap
from rasterio.features import rasterize
from scipy import ndimage

import config

VEG_MIN_HEIGHT_M = 2.5
VEG_MAX_HEIGHT_M = 50.0
HEIGHT_DECIMALS = 1  # heights in the outputs, m: 1 m LiDAR doesn't support centimetres
BUILDING_BUFFER_M = 1  # pixels at 1 m
STRUCT_NDVI = 0.5
STRUCT_ROUGH_M = 1.0
OBJECT_NDVI = 0.4
HEIGHT_BINS = [2.5, 5, 10, 15, 20, 30, np.inf]

OUT_DIR = config.OUTPUTS / "canopy"
BUILDINGS = config.OUTPUTS / "footprints" / "lidar_classified.gpkg"
BUILDING_MASK = config.OUTPUTS / "footprints" / "lidar_classified_mask.tif"
NDVI_PATH = config.PROCESSED / "ndvi_s2_2022.tif"
STRUCTURES = config.DATA / "osm" / "structures.gpkg"
SLIVER_M = 1.5
# extra width each side of a mapped bridge when clearing vegetation: OSM lines can sit a
# few metres off the deck (an M5 footbridge: 3-4 m), and narrow decks are too rough at
# their edges for the smooth rule. Buildings are cut at the mapped width only.
BRIDGE_TOLERANCE_M = {"foot": 3.5, "rail": 3.5, "road": 2.0}
MOSTLY_BRIDGE_SHARE = 0.25  # footprints this much covered by mapped structures are deleted
BUILDING, VEGETATION = 1, 2
SMOOTH_RULE, OBJECT_RULE, HEIGHT_CAP, MAPPED_BRIDGE = 1, 2, 3, 4

# 5 km quadrants as (row slice, col slice) on the 10,000 x 10,000 grid (row 0 = north)
QUADRANTS = {
    "SX99NW": (slice(0, 5000), slice(0, 5000)),
    "SX99NE": (slice(0, 5000), slice(5000, 10000)),
    "SX99SW": (slice(5000, 10000), slice(0, 5000)),
    "SX99SE": (slice(5000, 10000), slice(5000, 10000)),
    "all": (slice(None), slice(None)),
}
# preview windows: lower-left corner (BNG), 500 m square
PREVIEWS = {
    "exeter_centre": (292000, 92300),
    "rural_ne": (297500, 97500),
    "bridge_e": (296620, 92900),
    "structures_w": (291400, 92450),
    "st_thomas_viaduct": (291220, 91650),
    "footbridges_m5": (296650, 93270),
    "footbridge_m5_north": (297120, 95330),
}


def osm_structures():
    """Mapped bridges for vegetation (widened), every structure at its mapped width (for
    deleting bridge-like footprints) and the railway structures cut out of buildings."""
    if not STRUCTURES.exists():
        print(f"no {STRUCTURES}: run prepare_osm_structures.py; bridges and viaducts not removed")
        return None, None, None
    outlines = gpd.read_file(STRUCTURES, layer="bridge_outlines").geometry
    ways = gpd.read_file(STRUCTURES, layer="bridge_ways")
    decks = shapely.buffer(ways.geometry.values, ways["width_m"].values / 2, cap_style="flat")
    tolerance = ways["kind"].map(BRIDGE_TOLERANCE_M).values
    wide = shapely.buffer(ways.geometry.values, ways["width_m"].values / 2 + tolerance, cap_style="flat")
    bridges = shapely.union_all(np.concatenate([outlines.values, wide]))

    rail = ways["kind"].values == "rail"
    platforms = gpd.read_file(STRUCTURES, layer="platforms")

    def raised(tags):
        try:
            return float(json.loads(tags).get("layer", "0").split(";")[0]) >= 1
        except ValueError:
            return False

    plat = platforms[platforms["tags"].map(raised)].geometry
    plat = [g if g.geom_type == "Polygon" else g.buffer(2.0, cap_style="flat") for g in plat]
    plat = np.array(plat, dtype=object)
    cut = shapely.union_all(np.concatenate([decks[rail], plat]))
    structures = shapely.union_all(np.concatenate([outlines.values, decks, plat]))
    return bridges, structures, cut


def drop_bridges(fp, structures):
    """Delete footprints that are mostly mapped structure: bridges the classifier took for buildings."""
    hit = fp.intersects(structures)
    share = pd.Series(0.0, index=fp.index)
    share[hit] = shapely.area(shapely.intersection(fp.geometry[hit].values, structures)) / fp.geometry[hit].area
    drop = share >= MOSTLY_BRIDGE_SHARE
    return fp[~drop], int(drop.sum())


def cut_buildings(fp, cut):
    """Cut railway structures out of the footprints; recompute what changed."""
    fp = fp.assign(building_id=fp.index, cut_by_rail=False)
    hit = fp.intersects(cut)
    if not hit.any():
        return fp, 0
    pieces = fp[hit].copy()
    pieces["geometry"] = pieces.geometry.difference(cut)
    # opening (shrink then grow) removes slivers; mitre joins keep corners square
    pieces["geometry"] = pieces.geometry.buffer(-SLIVER_M, join_style="mitre").buffer(SLIVER_M, join_style="mitre")
    pieces = pieces.explode(index_parts=False)
    pieces = pieces[pieces.geometry.area >= config.MIN_BUILDING_M2].reset_index(drop=True)
    pieces["cut_by_rail"] = True
    pieces["area_m2"] = pieces.geometry.area.round(1)
    pieces["vertices"] = shapely.get_num_coordinates(pieces.geometry.values)
    heights = importlib.import_module("10_building_heights").footprint_heights(pieces)
    pieces[heights.columns] = heights
    out = pd.concat([fp[~hit], pieces], ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=fp.crs), int(hit.sum())


def ndvi_1m(ndvi_10m):
    """Sentinel-2 NDVI bilinear-interpolated to the 1 m grid (gaps filled with the median)."""
    filled = np.where(np.isnan(ndvi_10m), np.nanmedian(ndvi_10m), ndvi_10m)
    return ndimage.zoom(filled, 10, order=1).astype(np.float32)


def roughness(height):
    """Standard deviation of height in each 3x3 window (m): high in tree canopy, low on decks."""
    mean = ndimage.uniform_filter(height, 3)
    mean_sq = ndimage.uniform_filter(height * height, 3)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0)).astype(np.float32)


def classify(ndsm, valid, building, footprint_pixels, ndvi, bridge_pixels):
    """Classes (building, vegetation), canopy heights and the removed-structure codes.

    building: the building mask; footprint_pixels: every pixel a footprint polygon touches;
    bridge_pixels: every pixel a mapped bridge touches.
    """
    height = np.where(valid, ndsm, 0).astype(np.float32)
    near_building = ndimage.binary_dilation(building, iterations=BUILDING_BUFFER_M) if BUILDING_BUFFER_M else building
    near_building |= footprint_pixels
    tall = valid & (height > VEG_MIN_HEIGHT_M) & ~near_building

    removed = np.zeros(height.shape, np.uint8)
    smooth = tall & (ndvi < STRUCT_NDVI) & (roughness(height) < STRUCT_ROUGH_M)
    removed[smooth] = SMOOTH_RULE
    veg = tall & ~smooth

    labels, n = ndimage.label(veg, structure=np.ones((3, 3)))
    size = np.bincount(labels.ravel(), minlength=n + 1)
    mean_ndvi = np.bincount(labels.ravel(), weights=ndvi.ravel(), minlength=n + 1) / np.maximum(size, 1)
    not_green = mean_ndvi < OBJECT_NDVI
    not_green[0] = False
    object_rule = not_green[labels]
    removed[object_rule] = OBJECT_RULE
    veg &= ~object_rule
    noise = veg & (height > VEG_MAX_HEIGHT_M)
    removed[noise] = HEIGHT_CAP
    veg &= ~noise
    mapped = veg & bridge_pixels
    removed[mapped] = MAPPED_BRIDGE
    veg &= ~mapped

    classes = np.zeros(height.shape, np.uint8)
    classes[valid & building] = BUILDING  # for area statistics; touched pixels only clip vegetation
    classes[veg] = VEGETATION
    chm = np.where(veg, height, 0).astype(np.float32)
    chm[~valid] = config.NODATA
    return classes, chm, removed


def summarise(classes, chm, removed, valid):
    rows = []
    for name, (r, c) in QUADRANTS.items():
        cl, h, rm, ok = classes[r, c], chm[r, c], removed[r, c], valid[r, c]
        veg = cl == VEGETATION
        tall_other = veg.sum() + (rm > 0).sum()
        heights = h[veg]
        row = {
            "quadrant": name,
            "area_km2": ok.sum() / 1e6,
            "building_pct": 100 * (cl == BUILDING).sum() / ok.sum(),
            "vegetation_pct": 100 * veg.sum() / ok.sum(),
            "removed_smooth_pct_of_tall": 100 * (rm == SMOOTH_RULE).sum() / max(tall_other, 1),
            "removed_object_pct_of_tall": 100 * (rm == OBJECT_RULE).sum() / max(tall_other, 1),
            "removed_above_cap_px": int((rm == HEIGHT_CAP).sum()),
            "removed_mapped_bridge_px": int((rm == MAPPED_BRIDGE).sum()),
            "height_median_m": float(np.median(heights)) if heights.size else np.nan,
            "height_p95_m": float(np.percentile(heights, 95)) if heights.size else np.nan,
            "height_p999_m": float(np.percentile(heights, 99.9)) if heights.size else np.nan,
            "height_max_m": float(heights.max()) if heights.size else np.nan,
        }
        counts, _ = np.histogram(heights, HEIGHT_BINS)
        for lo, hi, n in zip(HEIGHT_BINS[:-1], HEIGHT_BINS[1:], counts, strict=True):
            label = f"{lo:g}+" if np.isinf(hi) else f"{lo:g}-{hi:g}"
            row[f"veg_{label}_m_pct"] = 100 * n / max(heights.size, 1)
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def write(path, array, profile, **kw):
    p = profile | {"compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512} | kw
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array, 1)


def preview(name, x0, y0, classes, chm, removed, transform, footprints):
    col0, row0 = ~transform * (x0, y0 + 500)
    r, c = slice(int(row0), int(row0) + 500), slice(int(col0), int(col0) + 500)
    cl, rm = classes[r, c], removed[r, c]
    h = np.ma.masked_where(chm[r, c] <= 0, chm[r, c])
    extent = (x0, x0 + 500, y0, y0 + 500)
    window = footprints.cx[x0:x0 + 500, y0:y0 + 500]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    veg = np.where(cl == VEGETATION, 1, 0)
    axes[0].imshow(veg, cmap=ListedColormap(["#f2f2f2", "#3a8f3a"]), vmin=0, vmax=1, extent=extent,
                   interpolation="nearest")
    axes[0].imshow(np.ma.masked_where(rm == 0, rm),
                   cmap=ListedColormap(["#d62728", "#1f77b4", "#ff7f0e", "#9467bd"]),
                   vmin=1, vmax=4, extent=extent, interpolation="nearest")
    window.plot(ax=axes[0], facecolor="#7a7a7a", edgecolor="black", linewidth=0.4)
    axes[0].set_title("Buildings grey (vectors), vegetation green; removed: smooth red,\n"
                      "object blue, >50 m orange, mapped bridge purple", fontsize=9)

    window.plot(ax=axes[1], facecolor="#7a7a7a", edgecolor="none")
    im = axes[1].imshow(h, cmap="YlGn", vmin=2.5, vmax=25, extent=extent, interpolation="nearest")
    fig.colorbar(im, ax=axes[1], shrink=0.75, label="canopy height (m)")
    axes[1].set_title("Canopy height (buildings grey)", fontsize=10)
    for ax in axes:
        ax.set_xlim(x0, x0 + 500)
        ax.set_ylim(y0, y0 + 500)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=8)
    fig.suptitle(f"{name}: 500 m window at E {x0}, N {y0}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"preview_{name}.png", dpi=110)
    plt.close(fig)


def write_style(path, max_height):
    """QGIS style for the height raster: green ramp from VEG_MIN_HEIGHT_M, read on load."""
    stops = [(VEG_MIN_HEIGHT_M, "#f7fcb9"), (10, "#addd8e"), (20, "#41ab5d"), (max_height, "#004529")]
    items = "".join(
        f'<item alpha="255" value="{v:g}" label="{v:g} m" color="{c}"/>' for v, c in stops
    )
    path.write_text(
        '<!DOCTYPE qgis><qgis version="3.34" styleCategories="Symbology"><pipe>'
        '<rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" '
        f'classificationMin="{VEG_MIN_HEIGHT_M:g}" classificationMax="{max_height:g}">'
        '<rastershader><colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0">'
        f"{items}</colorrampshader></rastershader></rasterrenderer></pipe></qgis>",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with rasterio.open(config.NDSM_PATH) as src:
        ndsm, profile, transform = src.read(1), src.profile, src.transform
    with rasterio.open(BUILDING_MASK) as src:
        building = src.read(1) > 0
    with rasterio.open(NDVI_PATH) as src:
        ndvi = ndvi_1m(src.read(1, masked=True).filled(np.nan))
    valid = ndsm != config.NODATA
    footprints = gpd.read_file(BUILDINGS, fid_as_index=True)
    bridges, structures, rail_cut = osm_structures()
    if structures is not None:
        footprints, n_drop = drop_bridges(footprints, structures)
        footprints, n_cut = cut_buildings(footprints, rail_cut)
        print(f"{n_drop} footprints deleted as mapped bridges, {n_cut} cut by railway viaducts, "
              f"bridges or raised platforms")
    else:
        footprints = footprints.assign(building_id=footprints.index, cut_by_rail=False)
    footprints["source"] = np.where(footprints["cut_by_rail"], "cut_by_rail", "classified")

    def touched(geoms):  # every pixel a geometry touches, even at its edge
        if geoms is None or len(geoms) == 0:
            return np.zeros(ndsm.shape, bool)
        return rasterize(((g, 1) for g in geoms), out_shape=ndsm.shape, transform=transform,
                         all_touched=True, dtype="uint8").astype(bool)

    footprint_pixels = touched(footprints.geometry)  # so vegetation never overlaps a building
    bridge_pixels = touched(None if bridges is None else shapely.get_parts(bridges))

    classes, chm, removed = classify(ndsm, valid, building, footprint_pixels, ndvi, bridge_pixels)

    overlap = int(((chm > 0) & footprint_pixels).sum())
    assert overlap == 0, f"{overlap} vegetation pixels touch a building footprint"

    height_cols = ["height_median_m", "height_p95_m", "height_max_m"]
    footprints[height_cols] = footprints[height_cols].astype(float).round(HEIGHT_DECIMALS)
    footprints.to_file(OUT_DIR / "buildings.gpkg", layer="buildings", driver="GPKG")
    vegetation = np.where(chm > 0, np.round(chm, HEIGHT_DECIMALS), config.NODATA).astype(np.float32)
    write(OUT_DIR / "vegetation_height.tif", vegetation, profile, dtype="float32", nodata=config.NODATA)
    write_style(OUT_DIR / "vegetation_height.qml", VEG_MAX_HEIGHT_M)
    write(OUT_DIR / "removed.tif", removed, profile, dtype="uint8", nodata=None)
    # earlier outputs of this step
    for old in ("classes.tif", "chm.tif", "preview_terraces_heavitree.png", "preview_houses_alphington.png",
                "preview_low_building_291278.png"):
        (OUT_DIR / old).unlink(missing_ok=True)
    (config.OUTPUTS / "eval" / "recovery.csv").unlink(missing_ok=True)
    (config.OUTPUTS / "diagnostics" / "recovery_objects.parquet").unlink(missing_ok=True)
    print(f"{len(footprints):,} buildings, {int((chm > 0).sum()):,} vegetation pixels, "
          f"0 pixels in both")

    summary = summarise(classes, chm, removed, valid)
    summary.to_csv(config.OUTPUTS / "eval" / "canopy_summary.csv", index=False)
    print(summary.T.to_string(header=False))
    for name, (x0, y0) in PREVIEWS.items():
        preview(name, x0, y0, classes, chm, removed, transform, footprints)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
