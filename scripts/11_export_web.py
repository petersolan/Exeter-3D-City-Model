"""Export the 3D city model for the web viewer (site/).

Writes to site/data/:
  buildings.pmtiles   final buildings (12_canopy_height.py): footprints with heights and
                      source; outlines simplified by 0.5 m in British National Grid
  trees.pmtiles       tree crowns with heights and single-pixel holes closed (13_tree_crowns.py),
                      simplified by 0.5 m
  rail.geojson        railway lines, with viaducts and bridges flagged (OpenStreetMap,
                      prepare_osm_structures.py)
  terrain/{z}/{x}/{y}.png   the LiDAR DTM as terrain-RGB tiles (Mapbox encoding,
                      0.1 m steps) at TERRAIN_RES_M, for MapLibre's 3D terrain and hillshade
  stats.json          headline numbers shown on the page
Vector tiles are written by GDAL's PMTiles driver (one static file each, read in the
browser with the pmtiles library). Only LiDAR-derived and open data are exported;
the OS ground truth never is.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

import config

SITE = config.ROOT / "site" / "data"
CANOPY = config.OUTPUTS / "canopy"
BUILDINGS = CANOPY / "buildings.gpkg"
TREES = CANOPY / "tree_crowns_filled.gpkg"  # single-pixel holes closed (13_tree_crowns.py)
RAIL = config.DATA / "osm" / "structures.gpkg"
TERRAIN_RES_M = 5.0
TERRAIN_ZOOMS = "10-15"
SIMPLIFY_M = 0.5


def to_pmtiles(gdf, out, layer, minzoom, maxzoom):
    """GeoDataFrame (WGS84) -> one-layer vector PMTiles via GDAL."""
    out.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"{layer}.gpkg"
        gdf.to_file(src, layer=layer, driver="GPKG")
        subprocess.run(["ogr2ogr", "-f", "PMTiles", str(out), str(src), "-nln", layer,
                        "-dsco", f"MINZOOM={minzoom}", "-dsco", f"MAXZOOM={maxzoom}"], check=True)
    print(f"{len(gdf):,} {layer} -> {out.name} ({out.stat().st_size / 1e6:.1f} MB)")


def prepare(path, columns, layer=None):
    gdf = gpd.read_file(path, layer=layer)[columns + ["geometry"]]
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    return gdf.to_crs("EPSG:4326")


def terrain_tiles(out_dir):
    """DTM -> Web Mercator at TERRAIN_RES_M -> terrain-RGB GeoTIFF -> XYZ PNG tiles."""
    with rasterio.open(config.DTM_PATH) as src:
        transform, w, h = calculate_default_transform(src.crs, "EPSG:3857", src.width, src.height,
                                                      *src.bounds, resolution=TERRAIN_RES_M)
        dem = np.full((h, w), np.nan, np.float32)
        reproject(rasterio.band(src, 1), dem, dst_transform=transform, dst_crs="EPSG:3857",
                  resampling=Resampling.bilinear, src_nodata=src.nodata, dst_nodata=np.nan)
    dem = np.where(np.isnan(dem), 0, dem)  # outside the AOI: sea level (the view is kept inside)
    code = np.round((dem + 10000) / 0.1).astype(np.uint32)
    rgb = np.stack([(code >> 16) & 255, (code >> 8) & 255, code & 255]).astype(np.uint8)
    shutil.rmtree(out_dir, ignore_errors=True)
    with tempfile.TemporaryDirectory() as tmp:
        tif = Path(tmp) / "terrain_rgb.tif"
        profile = {"driver": "GTiff", "width": w, "height": h, "count": 3, "dtype": "uint8",
                   "crs": "EPSG:3857", "transform": transform}
        with rasterio.open(tif, "w", **profile) as dst:
            dst.write(rgb)
        # nearest neighbour only: averaging encoded RGB values would corrupt the heights
        gdal2tiles = Path(sys.executable).parent / "Scripts" / "gdal2tiles.py"
        cmd = [sys.executable, str(gdal2tiles)] if gdal2tiles.exists() else ["gdal2tiles"]
        subprocess.run(cmd + ["--xyz", "-z", TERRAIN_ZOOMS, "-r", "near", "-w", "none", "--processes=4",
                              "-q", str(tif), str(out_dir)], check=True)
    n = sum(1 for _ in out_dir.rglob("*.png"))
    size = sum(p.stat().st_size for p in out_dir.rglob("*.png")) / 1e6
    print(f"terrain: {n:,} tiles ({size:.1f} MB) at {TERRAIN_RES_M:g} m -> {out_dir}")


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "footprints.geojson").unlink(missing_ok=True)  # replaced by buildings.pmtiles

    b = prepare(BUILDINGS, ["area_m2", "height_median_m", "height_p95_m", "height_max_m", "source"])
    b = b.dropna(subset=["height_max_m"])
    to_pmtiles(b, SITE / "buildings.pmtiles", "buildings", 12, 16)

    t = prepare(TREES, ["height_max_m", "height_p95_m", "crown_diameter_m"])
    to_pmtiles(t, SITE / "trees.pmtiles", "trees", 13, 16)

    rail = gpd.read_file(RAIL, layer="railway")[["railway", "bridge", "geometry"]]
    rail["on_bridge"] = rail["bridge"] != "no"
    rail = rail.clip(config.ANALYSIS_BOUNDS)
    rail.to_crs("EPSG:4326").to_file(SITE / "rail.geojson", driver="GeoJSON",
                                     layer_options={"COORDINATE_PRECISION": 6, "RFC7946": "YES"})

    terrain_tiles(SITE / "terrain")

    summary_path = config.OUTPUTS / "eval" / "summary.csv"
    summary = pd.read_csv(summary_path).set_index("dataset")
    # the final buildings' score only if it was evaluated after they were last written
    current = summary_path.stat().st_mtime > BUILDINGS.stat().st_mtime
    final = summary.loc["lidar_final"] if current and "lidar_final" in summary.index else summary.loc["lidar"]
    heights = pd.read_csv(config.OUTPUTS / "eval" / "heights.csv")
    stats = {"buildings": int(len(b)), "trees": int(len(t)), "area_f1": round(float(final["f1"]), 3),
             "height_within_2m_pct": round(float(heights.iloc[0].filter(like="within_2").iloc[0]), 0)}
    (SITE / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print("stats:", stats)


if __name__ == "__main__":
    main()
