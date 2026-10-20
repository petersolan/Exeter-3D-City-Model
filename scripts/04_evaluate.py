"""Evaluate footprint datasets against the OS ground truth.

Datasets: the LiDAR method's output (outputs/footprints/lidar_classified.gpkg),
the final buildings after 12_canopy_height.py (outputs/canopy/buildings.gpkg) and
the three free benchmarks (Microsoft, OpenStreetMap, OS Open). Everything
is computed on vector geometry within the analysis AOI.

Ground truth buildings filtered out by groundtruth.load_ground_truth (excluded
types, under 30 m2, below 3.5 m, no height) become ignore zones: they are
removed from the scoring area, and predicted shapes lying mostly inside them are
not counted. Predicted footprints under 30 m2 are left out for every dataset.

Metrics, each overall and per 5 km quadrant (SW and SE urban, NW and NE
rural). The LiDAR footprints in each quadrant come from a classifier that was
trained without it (leave-one-quadrant-out, see 09_object_classifier.py):

1. Area      precision, recall, F1 and IoU of the dissolved footprint area: how
             much of the mapped area is right. Unaffected by how a dataset
             splits buildings, so this is the headline comparison.
2. Object    ground truth and predictions are dissolved into blocks (touching
             buildings merge, e.g. a terrace becomes one block) and matched one
             to one at IoU >= 0.5: are the shapes right, not just the area?
3. Building  detection rate: share of ground truth buildings at least 50%
             covered (building-level recall). Footprint precision: share of
             predicted footprints with at least 50% of their area on scored
             ground truth (building-level precision; footprints mostly in ignore
             zones are left out). Detection F1 combines the two. Detection is
             also broken down by building class and size.

Uncertainty: 95% intervals from a block bootstrap over the 100 1 km tiles of
the AOI (tiles are resampled, so spatially clustered errors are respected).
The same resamples give paired comparisons of the LiDAR method against each
benchmark: the F1 difference with its interval, and the share of resamples in
which LiDAR scores higher.

Writes to outputs/eval/: summary.csv (overall), by_quadrant.csv, comparison.csv,
by_class.csv, by_size.csv and gt_coverage.gpkg (per-building coverage).
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

import config
from groundtruth import load_ground_truth

EVAL_DIR = config.OUTPUTS / "eval"
# lidar: the classifier's footprints (09 + 08); lidar_final: after 12_canopy_height.py
# (railway viaducts cut out, missed buildings recovered among the tall non-building pixels)
DATASETS = {"lidar": config.OUTPUTS / "footprints" / "lidar_classified.gpkg",
            "lidar_final": config.OUTPUTS / "canopy" / "buildings.gpkg", **config.BENCHMARKS}

minx, miny, maxx, maxy = config.ANALYSIS_BOUNDS
midx, midy = (minx + maxx) / 2, (miny + maxy) / 2
QUADRANTS = {
    "sw (urban)": box(minx, miny, midx, midy),
    "se (urban)": box(midx, miny, maxx, midy),
    "nw (rural)": box(minx, midy, midx, maxy),
    "ne (rural)": box(midx, midy, maxx, maxy),
}
TILE_M = 1000
N_BOOT = 2000
MATCH_SHARE = 0.5  # coverage / overlap needed for a building or footprint to count

CLASSES = {
    "terraced/semi house": ["Mid-Terrace House", "End-Of-Terrace House", "Semi-Detached House"],
    "detached house": ["Detached House"],
    "flats/other residential": ["Multiple Residential Accommodation", "Residential Building",
                                "Static Caravan Or Mobile Home",
                                "Temporary Or Holiday Accommodation Building"],
    "outbuilding/unknown": ["Domestic Outbuilding", "Unknown Building", "Ancillary Building"],
    "commercial/mixed use": ["Commercial Building", "Mixed Use Building"],
    "small utility": ["Electricity Sub Station", "Gas Governor", "Pumping Station",
                      "Utility Building", "Waste Water Treatment Works",
                      "Water Distribution Facility", "Electricity Distribution Facility"],
}
SIZE_BINS = [config.MIN_BUILDING_M2, 50, 200, 1000, np.inf]
SIZE_LABELS = ["30-50", "50-200", "200-1000", ">1000"]


def polygons(geoms):
    """Valid, single-part polygons from any geometry array."""
    parts = shapely.get_parts(shapely.make_valid(np.asarray(geoms)))
    parts = parts[shapely.get_type_id(parts) == 3]  # Polygon
    return parts[shapely.area(parts) > 0]


def load(path, aoi):
    gdf = gpd.read_parquet(path) if path.suffix == ".parquet" else gpd.read_file(path)
    if gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs(config.CRS)
    gdf = gdf[gdf.intersects(aoi)]
    # Too small to count as a building, for every dataset (ground truth too)
    return gdf[gdf.area >= config.MIN_BUILDING_M2]


def blocks(geoms):
    """Dissolve to non-overlapping blocks; touching polygons merge."""
    return polygons([shapely.union_all(polygons(geoms))])


def prf(tp, pred, truth):
    """Precision, recall and F1 from a true-positive amount and the two totals."""
    precision, recall = tp / pred if pred else 0.0, tp / truth if truth else 0.0
    return precision, recall, 2 * precision * recall / (precision + recall) if tp else 0.0


def area_metrics(gt_u, pr_u, region):
    g = shapely.intersection(gt_u, region)
    p = shapely.intersection(pr_u, region)
    tp = shapely.intersection(g, p).area
    precision, recall, f1 = prf(tp, p.area, g.area)
    return dict(precision=precision, recall=recall, f1=f1,
                iou=tp / shapely.union(g, p).area,
                gt_km2=g.area / 1e6, pred_km2=p.area / 1e6)


def overlap_fraction(geoms, others):
    """Share of each geometry's area covered by `others` (non-overlapping polygons)."""
    i, j = shapely.STRtree(others).query(geoms, predicate="intersects")
    covered = np.bincount(i, shapely.area(shapely.intersection(geoms[i], others[j])), len(geoms))
    return covered / shapely.area(geoms)


def in_region(geoms, region):
    """Assign shapes to a region by a point inside them (each shape counts once)."""
    return shapely.within(shapely.point_on_surface(geoms), region)


def object_metrics(gt_blocks, pr_blocks, ignore_parts, region=None):
    in_ignore = overlap_fraction(pr_blocks, ignore_parts)
    pr_b = pr_blocks[in_ignore <= MATCH_SHARE]
    gt_b = gt_blocks
    if region is not None:
        gt_b, pr_b = gt_b[in_region(gt_b, region)], pr_b[in_region(pr_b, region)]
    gt = gpd.GeoDataFrame(geometry=gt_b, crs=config.CRS)
    pr = gpd.GeoDataFrame(geometry=pr_b, crs=config.CRS)
    pairs = gpd.sjoin(gt, pr, predicate="intersects")
    g = gt.geometry.values[pairs.index.values]
    p = pr.geometry.values[pairs["index_right"].values]
    iou = shapely.area(shapely.intersection(g, p)) / shapely.area(shapely.union(g, p))
    # IoU >= 0.5 implies a one-to-one match between non-overlapping blocks
    tp = pairs[iou >= 0.5].index.nunique()
    precision, recall, f1 = prf(tp, len(pr), len(gt))
    return dict(gt_blocks=len(gt), pred_blocks=len(pr), matched=tp,
                obj_precision=precision, obj_recall=recall, obj_f1=f1)


def coverage(gt, pr_blocks):
    """Fraction of each ground truth building's area covered by the prediction."""
    pr = gpd.GeoDataFrame(geometry=pr_blocks, crs=config.CRS)
    pieces = gpd.overlay(gt[["gt_id", "geometry"]], pr, how="intersection", keep_geom_type=True)
    covered = pieces.assign(a=pieces.area).groupby("gt_id")["a"].sum()
    return (covered.reindex(gt["gt_id"]).fillna(0).values / gt.area.values).clip(0, 1)


def building_metrics(detected, fp_correct):
    """Detection rate (building recall), footprint precision and their F1, in %."""
    det = 100 * np.mean(detected) if len(detected) else np.nan
    prec = 100 * np.mean(fp_correct) if len(fp_correct) else np.nan
    return dict(bld_detected_pct=det, footprint_precision_pct=prec,
                detection_f1_pct=2 * det * prec / (det + prec) if det + prec else 0.0)


def tile_grid():
    xs = np.arange(minx, maxx, TILE_M)
    ys = np.arange(miny, maxy, TILE_M)
    return np.array([box(x, y, x + TILE_M, y + TILE_M) for x in xs for y in ys])


def bootstrap_index(n_tiles, seed=0):
    return np.random.default_rng(seed).integers(0, n_tiles, size=(N_BOOT, n_tiles))


def ci(values):
    return np.nanpercentile(values, [2.5, 97.5])


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    aoi = gpd.read_file(config.AOI_ANALYSIS_PATH).union_all()

    gt, ignored = load_ground_truth(aoi)
    gt["gt_id"] = np.arange(len(gt))
    lookup = {d: c for c, ds in CLASSES.items() for d in ds}
    gt["class"] = gt["description"].map(lookup).fillna("public/other")
    gt["size_band"] = pd.cut(gt.area, SIZE_BINS, labels=SIZE_LABELS, right=False)
    gt_blocks = blocks(shapely.intersection(gt.geometry.values, aoi))
    gt_u = shapely.union_all(gt_blocks)
    ignore_u = shapely.difference(shapely.union_all(polygons(ignored.geometry.values)), gt_u)
    ignore_parts = polygons([ignore_u])
    region = shapely.difference(aoi, ignore_u)
    quadrants = {q: shapely.intersection(b, region) for q, b in QUADRANTS.items()}
    gt_points = shapely.point_on_surface(gt.geometry.values)
    gt_quadrant = {q: shapely.within(gt_points, b) for q, b in QUADRANTS.items()}
    print(f"Ground truth: {len(gt):,} scored buildings -> {len(gt_blocks):,} blocks; "
          f"{len(ignored):,} ignored ({ignore_u.area / 1e6:.3f} km2)")

    # Per-tile areas and counts for the bootstrap; the scoring region is fixed per tile
    tiles = tile_grid()
    gt_r = shapely.intersection(gt_u, region)
    tile_gt = shapely.area(shapely.intersection(gt_r, tiles))
    # Tile of each building, in tile_grid() order (x columns, then y within each)
    n_y = int((maxy - miny) // TILE_M)
    ix = np.clip((shapely.get_x(gt_points) - minx) // TILE_M, 0, n_y - 1).astype(int)
    iy = np.clip((shapely.get_y(gt_points) - miny) // TILE_M, 0, n_y - 1).astype(int)
    gt_tile = ix * n_y + iy
    boot = bootstrap_index(len(tiles))

    summary, quads, by_class, by_size, per_tile = [], [], [], [], {}
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue
        pr = load(path, aoi)
        geoms = polygons(shapely.intersection(pr.geometry.values, aoi))
        geoms = geoms[shapely.area(geoms) >= config.MIN_BUILDING_M2]  # no slivers at the AOI edge
        pr_blocks = blocks(geoms)
        pr_u = shapely.union_all(pr_blocks)

        cov = coverage(gt, pr_blocks)
        gt[f"cov_{name}"] = cov.round(3)
        detected = cov >= MATCH_SHARE
        # Footprint precision: individual footprints, leaving out those mostly in ignore zones
        counted = overlap_fraction(geoms, ignore_parts) <= MATCH_SHARE
        fp_geoms = geoms[counted]
        fp_correct = overlap_fraction(fp_geoms, gt_blocks) >= MATCH_SHARE

        row = dict(dataset=name, features=len(pr), **area_metrics(gt_u, pr_u, region),
                   **object_metrics(gt_blocks, pr_blocks, ignore_parts),
                   **building_metrics(detected, fp_correct))

        # Bootstrap over 1 km tiles: area TP / predicted / truth and detection per tile
        p_r = shapely.intersection(pr_u, region)
        t_pred = shapely.area(shapely.intersection(p_r, tiles))
        t_tp = shapely.area(shapely.intersection(shapely.intersection(gt_r, p_r), tiles))
        t_det = np.bincount(gt_tile, detected, len(tiles))
        t_n = np.bincount(gt_tile, minlength=len(tiles))
        tp_b, pred_b, gt_b = t_tp[boot].sum(1), t_pred[boot].sum(1), tile_gt[boot].sum(1)
        prec_b, rec_b = tp_b / pred_b, tp_b / gt_b
        f1_b = 2 * prec_b * rec_b / (prec_b + rec_b)
        det_b = 100 * t_det[boot].sum(1) / t_n[boot].sum(1)
        per_tile[name] = f1_b
        for key, vals in (("precision", prec_b), ("recall", rec_b), ("f1", f1_b),
                          ("bld_detected_pct", det_b)):
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = ci(vals)
        summary.append(row)
        print(f"{name}: area P {row['precision']:.3f} R {row['recall']:.3f} F1 {row['f1']:.3f} "
              f"[{row['f1_ci_low']:.3f}-{row['f1_ci_high']:.3f}], object F1 {row['obj_f1']:.3f}, "
              f"detected {row['bld_detected_pct']:.1f}%, footprint precision "
              f"{row['footprint_precision_pct']:.1f}%", flush=True)

        fp_points = shapely.point_on_surface(fp_geoms)
        for q, qregion in quadrants.items():
            quads.append(dict(dataset=name, quadrant=q, **area_metrics(gt_u, pr_u, qregion),
                              **object_metrics(gt_blocks, pr_blocks, ignore_parts, QUADRANTS[q]),
                              **building_metrics(detected[gt_quadrant[q]],
                                                 fp_correct[shapely.within(fp_points, QUADRANTS[q])])))

        det = gt.assign(detected=detected, covered_m2=cov * gt.area, gt_m2=gt.area)
        for key, out in (("class", by_class), ("size_band", by_size)):
            grp = det.groupby(key, observed=True)
            out.append(pd.DataFrame({
                "dataset": name,
                "buildings": grp.size(),
                "detected_pct": 100 * grp["detected"].mean(),
                "area_recall_pct": 100 * grp["covered_m2"].sum() / grp["gt_m2"].sum(),
            }).reset_index())

    # Paired comparison: LiDAR against each benchmark on the same tile resamples
    comparison = []
    for name in (n for n in per_tile if n != "lidar"):
        diff = per_tile["lidar"] - per_tile[name]
        base = {r["dataset"]: r["f1"] for r in summary}
        low, high = ci(diff)
        comparison.append(dict(benchmark=name, lidar_f1=base["lidar"], benchmark_f1=base[name],
                               f1_difference=base["lidar"] - base[name], ci_low=low, ci_high=high,
                               lidar_better_pct=100 * np.mean(diff > 0)))

    pd.DataFrame(summary).round(4).to_csv(EVAL_DIR / "summary.csv", index=False)
    pd.DataFrame(quads).round(4).to_csv(EVAL_DIR / "by_quadrant.csv", index=False)
    pd.DataFrame(comparison).round(4).to_csv(EVAL_DIR / "comparison.csv", index=False)
    pd.concat(by_class).round(1).to_csv(EVAL_DIR / "by_class.csv", index=False)
    pd.concat(by_size).round(1).to_csv(EVAL_DIR / "by_size.csv", index=False)
    gt.drop(columns="size_band").assign(size_band=gt["size_band"].astype(str)).to_file(
        EVAL_DIR / "gt_coverage.gpkg", driver="GPKG", layer="gt_coverage")

    pd.set_option("display.width", 250)
    cols = ["dataset", "features", "precision", "recall", "f1", "f1_ci_low", "f1_ci_high", "iou",
            "obj_precision", "obj_recall", "obj_f1", "bld_detected_pct", "footprint_precision_pct",
            "detection_f1_pct"]
    print("\n" + pd.DataFrame(summary)[cols].round(3).to_string(index=False))
    print("\nLiDAR vs benchmarks (area F1, paired tile bootstrap):\n"
          + pd.DataFrame(comparison).round(3).to_string(index=False))
    qdf = pd.DataFrame(quads)
    for metric in ("f1", "obj_f1", "bld_detected_pct", "footprint_precision_pct"):
        print(f"\n{metric} by quadrant\n"
              + qdf.pivot(index="quadrant", columns="dataset", values=metric).round(3).to_string())
    for title, frames, idx in (("Buildings detected % by class", by_class, "class"),
                               ("Buildings detected % by size (m2)", by_size, "size_band")):
        df = pd.concat(frames)
        print(f"\n{title}\n" + df.pivot(index=idx, columns="dataset", values="detected_pct")
              .round(1).to_string())


if __name__ == "__main__":
    main()
