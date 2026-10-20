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
Railway viaducts and bridges, and raised platforms, are cut out of the building
footprints: a viaduct at roof height otherwise merges with the buildings beside it
(e.g. St Thomas station, Exeter). Pieces left under MIN_BUILDING_M2 and slivers
narrower than 2 x SLIVER_M are dropped; cut pieces get new heights and keep the
original id in building_id.

Recovered buildings: houses the first extraction (03) never offered to the classifier
end up in the tall non-building pixels (vegetation, or removed by the smooth/object
rules), e.g. whole terraces. Roofs are planes, foliage is not: pixels whose 3x3
plane-fit residual is below RECOVER_RESIDUAL_M form flat cores, opened by 3x3 (gone
under 3 m wide: hedges, walls). Each core object gets size, flatness, greenness,
height, road and shape features, and its distance to and contact with the existing
footprints (missed houses sit in rows beside found ones). A gradient-boosting model,
leave-one-quadrant-out like 09, scores each quadrant with a model that never saw it;
objects with probability >= RECOVER_THRESHOLD (fixed, not tuned) are kept. Labels:
OS ground truth, as in 09. Kept cores grow RECOVER_GROW_PX pixel into
neighbouring tall pixels (eaves, roof edges), are traced to polygons with heights
(source = "recovered") and leave the vegetation layer.

Writes to outputs/canopy/:
  buildings.gpkg          building footprints (vector) with height columns; source is
                          classified (09), cut_by_rail (viaduct cut) or recovered
  vegetation_height.tif   height in m on vegetation pixels, no data elsewhere: click a
                          pixel (QGIS Identify) to read its height
  vegetation_height.qml   QGIS style (green height ramp), loaded with the raster
  removed.tif             tall non-building pixels taken out of vegetation: 1 smooth
                          rule, 2 object rule, 3 above the height cap, 4 mapped bridge
  preview_*.png           sample 500 m windows
and to outputs/eval/: canopy_summary.csv (overall and per 5 km quadrant) and
recovery.csv (recovered buildings per quadrant: threshold, objects, area, precision).
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
from rasterio.features import rasterize, shapes
from scipy import ndimage
from skimage import measure
from sklearn.ensemble import HistGradientBoostingClassifier

import config
from groundtruth import load_ground_truth

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
RECOVER_RESIDUAL_M = 0.2  # roof pixels: median plane residual 0.44 m, other tall pixels 1.16 m
RECOVER_GROW_PX = 1  # 1 px: 79% of the added area is building; 2 px: 76%, more is garden
# fixed, not tuned: tuning it on the other quadrants did not transfer between urban and
# rural ones (an 80% precision target gave 22-87% by quadrant)
RECOVER_THRESHOLD = 0.5
RECOVER_FEATURES = ["area", "core_frac", "ndvi", "h_mean", "h_std", "resid", "on_road",
                    "dist_bld", "touch_bld", "solidity", "elong"]
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
    "terraces_heavitree": (292810, 92680),
    "houses_alphington": (290330, 90900),
    "low_building_291278": (291030, 90750),
}


def osm_structures():
    """Mapped bridges (for vegetation) and the railway structures cut out of buildings."""
    if not STRUCTURES.exists():
        print(f"no {STRUCTURES}: run prepare_osm_structures.py; bridges and viaducts not removed")
        return None, None
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
    cut = shapely.union_all(np.concatenate([decks[rail], np.array(plat, dtype=object)]))
    return bridges, cut


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


def plane_residual(height):
    """RMS distance (m) from the best-fit plane in each 3x3 window: ~0 on any roof slope."""
    ones = np.ones((3, 3))
    kx = np.array([[-1.0, 0, 1]] * 3)

    def conv(a, k):
        return ndimage.convolve(a, k, mode="nearest")

    s, ss, sx, sy = conv(height, ones), conv(height * height, ones), conv(height, kx), conv(height, kx.T)
    rss = ss - s * s / 9 - sx * sx / 6 - sy * sy / 6  # x, y in {-1, 0, 1}: sum x^2 = 6
    return np.sqrt(np.maximum(rss, 0) / 9).astype(np.float32)


def recovery_objects(height, candidates, structure_px, ndvi, road, built, transform):
    """Roof-like objects among tall non-building pixels, with their features.

    Cores: flat pixels (plane residual < RECOVER_RESIDUAL_M) or pixels the smooth / object
    rules took out of vegetation (structure_px), opened by 2x2 so 1 px lines go. A small
    pitched roof is only ~3 m from ridge to eaves, so a 3x3 opening would erase it.
    Returns (core labels, features table indexed by label).
    """
    resid = plane_residual(height)
    core = candidates & ((resid < RECOVER_RESIDUAL_M) | structure_px)
    core = ndimage.binary_opening(core, structure=np.ones((2, 2)))
    labels, n = ndimage.label(core, structure=np.ones((3, 3)))
    if n == 0:
        return labels, pd.DataFrame()
    idx = labels.ravel()
    inside = idx > 0
    ids = idx[inside]
    area = np.bincount(ids, minlength=n + 1)[1:].astype(float)

    def mean(a):
        return np.bincount(ids, weights=a.ravel()[inside], minlength=n + 1)[1:] / area

    grown = ndimage.binary_dilation(core, iterations=1) & candidates
    dist = ndimage.distance_transform_edt(~built).astype(np.float32)
    ring = ndimage.grey_dilation(labels, size=(3, 3))
    ring = np.where(core, 0, ring).ravel()
    on_ring = ring > 0
    ring_len = np.bincount(ring[on_ring], minlength=n + 1)[1:].astype(float)
    ring_built = np.bincount(ring[on_ring], weights=built.ravel()[on_ring].astype(float), minlength=n + 1)[1:]
    props = pd.DataFrame(measure.regionprops_table(
        labels, properties=("label", "solidity", "axis_major_length", "axis_minor_length", "centroid")))
    h_mean = mean(height)
    f = pd.DataFrame({
        "area": area,
        "ndvi": mean(ndvi),
        "h_mean": h_mean,
        "h_std": np.sqrt(np.maximum(mean(height * height) - h_mean ** 2, 0)),
        "resid": mean(resid),
        "on_road": mean(road.astype(np.float32)),
        "dist_bld": mean(dist),
        "touch_bld": ring_built / np.maximum(ring_len, 1),
        "solidity": props["solidity"].values,
        "elong": (props["axis_minor_length"] / props["axis_major_length"].replace(0, np.nan)).fillna(0).values,
    }, index=np.arange(1, n + 1))
    # share of the object's 1 px-grown extent that is flat core (roofs are mostly core)
    grown_labels = ndimage.grey_dilation(labels, size=(3, 3)) * grown
    f["core_frac"] = area / np.maximum(np.bincount(grown_labels.ravel(), minlength=n + 1)[1:], 1)
    x, y = transform * (props["centroid-1"].values, props["centroid-0"].values)
    f["x"], f["y"] = x, y
    f["quadrant"] = np.where(f["y"] >= 95000, "N", "S") + np.where(f["x"] >= 295000, "E", "W")
    return labels, f


def fit(train):
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
    return model.fit(train[RECOVER_FEATURES], train["is_building"])


def score_recovery(f, scored_px, ignored_px, labels):
    """Labels from OS, leave-one-quadrant-out probabilities and per-quadrant thresholds."""
    idx = labels.ravel()
    inside = idx > 0
    n = len(f)

    def share(mask):
        return np.bincount(idx[inside], weights=mask.ravel()[inside].astype(float), minlength=n + 1)[1:] / f["area"].values

    f["scored"], f["ignored"] = share(scored_px), share(ignored_px)
    outside = 1 - f["scored"] - f["ignored"]
    f["is_building"] = (f["scored"] > outside).astype(int)  # the same rule as 09
    f["trainable"] = (f["scored"] + outside) > 0.5           # mostly not in ignore zones
    f["prob"], f["threshold"] = np.nan, RECOVER_THRESHOLD
    for q in sorted(f["quadrant"].unique()):
        train = f[(f["quadrant"] != q) & f["trainable"]]
        test = f["quadrant"] == q
        f.loc[test, "prob"] = fit(train).predict_proba(f.loc[test, RECOVER_FEATURES])[:, 1]
    f["keep"] = f["prob"] >= RECOVER_THRESHOLD
    return f


def recover_buildings(height, candidates, structure_px, ndvi, road, built, transform):
    """Recovered building pixels, the per-object table, and the OS scored / ignored pixels."""
    scored, ignored = load_ground_truth(shapely.box(*config.ANALYSIS_BOUNDS))

    def burn(gdf):
        return rasterize(((g, 1) for g in gdf.geometry), out_shape=height.shape, transform=transform,
                         dtype="uint8").astype(bool)

    scored_px, ignored_px = burn(scored), burn(ignored)
    labels, f = recovery_objects(height, candidates, structure_px, ndvi, road, built, transform)
    if f.empty:
        return np.zeros(height.shape, bool), f, scored_px, ignored_px
    f = score_recovery(f, scored_px, ignored_px, labels)
    keep = np.zeros(len(f) + 1, bool)
    keep[f.index[f["keep"]]] = True  # MIN_BUILDING_M2 applies to the traced polygons, not cores:
    core = keep[labels]              # a pitched roof splits into two smaller planes at its ridge
    mask = core | (ndimage.binary_dilation(core, iterations=RECOVER_GROW_PX) & candidates)
    return mask, f, scored_px, ignored_px


def recovered_polygons(mask, transform, crs):
    """Trace recovered pixels to polygons; drop pieces under MIN_BUILDING_M2."""
    polys = [shapely.geometry.shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask, transform=transform)
             if v == 1]
    gdf = gpd.GeoDataFrame(geometry=polys, crs=crs)
    gdf = gdf[gdf.area >= config.MIN_BUILDING_M2].reset_index(drop=True)
    gdf["area_m2"] = gdf.area.round(1)
    gdf["vertices"] = shapely.get_num_coordinates(gdf.geometry.values)
    heights = importlib.import_module("10_building_heights").footprint_heights(gdf)
    gdf[heights.columns] = heights
    return gdf


def recovery_report(f, mask, scored_px, ignored_px):
    rows = []
    for q, (r, c) in QUADRANTS.items():
        sub = f if q == "all" else f[f["quadrant"] == q[-2:]]
        m, s, ig = mask[r, c], scored_px[r, c], ignored_px[r, c]
        true, false = int((m & s).sum()), int((m & ~s & ~ig).sum())
        rows.append({
            "quadrant": q,
            "threshold": sub["threshold"].iloc[0] if q != "all" and len(sub) else np.nan,
            "objects": len(sub), "kept": int(sub["keep"].sum()),
            "recovered_m2": int(m.sum()), "on_os_building_m2": true, "outside_os_m2": false,
            "precision_pct": round(100 * true / max(true + false, 1), 1),
        })
    return pd.DataFrame(rows)


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
    bridges, rail_cut = osm_structures()
    if rail_cut is not None:
        footprints, n_cut = cut_buildings(footprints, rail_cut)
        print(f"{n_cut} footprints cut by railway viaducts, bridges or raised platforms")

    def touched(geoms):  # every pixel a geometry touches, even at its edge
        if geoms is None or len(geoms) == 0:
            return np.zeros(ndsm.shape, bool)
        return rasterize(((g, 1) for g in geoms), out_shape=ndsm.shape, transform=transform,
                         all_touched=True, dtype="uint8").astype(bool)

    footprint_pixels = touched(footprints.geometry)  # so vegetation never overlaps a building
    bridge_pixels = touched(None if bridges is None else shapely.get_parts(bridges))

    classes, chm, removed = classify(ndsm, valid, building, footprint_pixels, ndvi, bridge_pixels)

    # recover buildings among the tall non-building pixels (see the module docstring)
    height = np.where(valid, ndsm, 0).astype(np.float32)
    roads = gpd.read_file(config.ROADS_PATH, bbox=config.ANALYSIS_BOUNDS).to_crs(config.CRS)
    road = touched(roads.geometry.buffer(3))
    structure_px = np.isin(removed, (SMOOTH_RULE, OBJECT_RULE))
    candidates = (chm > 0) | structure_px
    recovered, objects, scored_px, ignored_px = recover_buildings(
        height, candidates, structure_px, ndvi, road, footprint_pixels, transform)
    report = recovery_report(objects, recovered, scored_px, ignored_px)
    report.to_csv(config.OUTPUTS / "eval" / "recovery.csv", index=False)
    objects.to_parquet(config.OUTPUTS / "diagnostics" / "recovery_objects.parquet")
    print("recovered buildings:\n" + report.to_string(index=False))

    footprints["source"] = np.where(footprints["cut_by_rail"], "cut_by_rail", "classified")
    new = recovered_polygons(recovered, transform, footprints.crs)
    new["building_id"] = np.arange(len(new)) + int(footprints["building_id"].max()) + 1
    new["cut_by_rail"], new["source"], new["orientation"] = False, "recovered", "traced"
    footprints = gpd.GeoDataFrame(pd.concat([footprints, new], ignore_index=True), crs=footprints.crs)

    # the recovered buildings leave vegetation and the removed codes
    footprint_pixels = touched(footprints.geometry)
    chm[footprint_pixels & (chm > 0)] = 0
    removed[recovered] = 0
    classes[recovered] = BUILDING
    classes[(classes == VEGETATION) & ~(chm > 0)] = 0
    overlap = int(((chm > 0) & footprint_pixels).sum())
    assert overlap == 0, f"{overlap} vegetation pixels touch a building footprint"

    height_cols = ["height_median_m", "height_p95_m", "height_max_m"]
    footprints[height_cols] = footprints[height_cols].astype(float).round(HEIGHT_DECIMALS)
    footprints.to_file(OUT_DIR / "buildings.gpkg", layer="buildings", driver="GPKG")
    vegetation = np.where(chm > 0, np.round(chm, HEIGHT_DECIMALS), config.NODATA).astype(np.float32)
    write(OUT_DIR / "vegetation_height.tif", vegetation, profile, dtype="float32", nodata=config.NODATA)
    write_style(OUT_DIR / "vegetation_height.qml", VEG_MAX_HEIGHT_M)
    write(OUT_DIR / "removed.tif", removed, profile, dtype="uint8", nodata=None)
    for old in ("classes.tif", "chm.tif"):  # earlier outputs of this step
        (OUT_DIR / old).unlink(missing_ok=True)
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
