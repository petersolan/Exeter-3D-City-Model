"""Census 2021 population for the AOI as a 5 m raster.

Residents come from the companion project GB-Census-Population-Map (local folder
Census-distribution, read only): Census 2021 people placed on address points (UPRNs),
with points at the same location combined (location_population_points.parquet). Each
5 m cell holds the summed residents of the points inside it, the same method as that
project's 10 m and 25 m rasters, here on the AOI's grid so it lines up with the
vegetation raster (every 5 x 5 block of 1 m pixels is one cell).

Writes outputs/population/population_5m.tif: UInt16, 0 = no residents (nodata), EPSG:27700,
Cloud Optimized GeoTIFF. Point-based: people sit where their address point is, usually
inside the building, so the raster shows dots rather than whole building footprints.

    python 14_population_raster.py [--points <location_population_points.parquet>]
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin

import config

POINTS = config.ROOT.parent / "Census-distribution" / "data" / "processed" / "location_population_points.parquet"
OUT = config.OUTPUTS / "population" / "population_5m.tif"
CELL_M = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--points", type=Path, default=POINTS, help="location_population_points.parquet")
    args = parser.parse_args()

    minx, miny, maxx, maxy = config.ANALYSIS_BOUNDS
    pts = gpd.read_parquet(args.points, columns=["population", "geometry"], bbox=(minx, miny, maxx, maxy))
    x, y = pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()
    # half-open cells: a point on the east or south edge belongs to the next square
    inside = (x >= minx) & (x < maxx) & (y > miny) & (y <= maxy)
    x, y, people = x[inside], y[inside], pts["population"].to_numpy()[inside].astype(np.int64)

    width, height = int((maxx - minx) / CELL_M), int((maxy - miny) / CELL_M)
    col = ((x - minx) // CELL_M).astype(np.int64)
    row = ((maxy - y) // CELL_M).astype(np.int64)
    grid = np.bincount(row * width + col, weights=people, minlength=width * height).reshape(height, width)
    assert grid.max() <= np.iinfo("uint16").max, "a cell exceeds UInt16"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp.tif")
    profile = dict(driver="GTiff", width=width, height=height, count=1, dtype="uint16", nodata=0,
                   crs=config.CRS, transform=from_origin(minx, maxy, CELL_M, CELL_M),
                   tiled=True, blockxsize=512, blockysize=512, compress="zstd")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(grid.astype("uint16"), 1)
        dst.set_band_description(1, "Population")
    rio_copy(tmp, OUT, driver="COG", compress="ZSTD", overview_resampling="average")
    tmp.unlink()

    total, cells = int(grid.sum()), int((grid > 0).sum())
    assert total == int(people.sum()), "people lost while binning"
    print(f"{len(people):,} address points, {total:,} residents in {cells:,} populated {CELL_M} m cells "
          f"(max {int(grid.max())} in one cell) -> {OUT}")


if __name__ == "__main__":
    main()
