"""Download bridges, railways, platforms, power towers and lines for the AOI from OpenStreetMap (Overpass API).

OS open data has railway centrelines (Zoomstack) but no bridges, viaducts or
footbridges: OS Open Roads holds roads only and marks no bridges. OpenStreetMap
tags them (bridge=yes / viaduct on footways, cycleways, roads and railways), and
often maps the bridge outline itself (man_made=bridge). Used by 12_canopy_height.py
to take bridges out of the vegetation layer and railway viaducts out of building
footprints.

Writes data/osm/structures.gpkg, layers (EPSG:27700):
  bridge_outlines  man_made=bridge polygons
  bridge_ways      ways tagged as a bridge (kind, width_m: tagged width or a default)
  railway          railway lines (bridge flag; tunnels left out)
  platforms        railway platforms (polygons)
  towers           power towers and poles, masts and towers (points; kind = the power or
                   man_made value): tall standalone structures that look like trees
  power_lines      overhead power lines (power=line / minor_line)
Data (c) OpenStreetMap contributors, ODbL.

Usage: python prepare_osm_structures.py
"""

import json
import time

import geopandas as gpd
import pandas as pd
import requests
import shapely
from pyproj import Transformer

import config

OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
OUT = config.DATA / "osm" / "structures.gpkg"
MARGIN_M = 200
# deck widths when OSM has no width tag (m); a pixel half covered still counts as deck
DEFAULT_WIDTH_M = {"foot": 4.0, "rail": 8.0, "road": 10.0}
FOOT = {"footway", "path", "cycleway", "steps", "bridleway", "pedestrian"}
RAIL = {"rail", "light_rail", "narrow_gauge", "subway", "tram", "preserved", "disused", "abandoned"}


def bbox_wgs84():
    minx, miny, maxx, maxy = config.ANALYSIS_BOUNDS
    to_wgs = Transformer.from_crs(config.CRS, "EPSG:4326", always_xy=True)
    xs, ys = to_wgs.transform([minx - MARGIN_M, maxx + MARGIN_M], [miny - MARGIN_M, maxy + MARGIN_M])
    return f"{min(ys)},{min(xs)},{max(ys)},{max(xs)}"  # south, west, north, east


def query() -> list[dict]:
    bb = bbox_wgs84()
    q = f"""[out:json][timeout:180];
    (
      way["man_made"="bridge"]({bb});
      way["bridge"]["bridge"!="no"]({bb});
      way["railway"]({bb});
      way["railway"="platform"]({bb});
      way["public_transport"="platform"]["train"="yes"]({bb});
      node["power"~"^(tower|pole|portal)$"]({bb});
      node["man_made"~"^(mast|tower|communications_tower)$"]({bb});
      way["power"~"^(line|minor_line)$"]({bb});
    );
    out tags geom;"""
    headers = {"User-Agent": "LiDAR-Building-Footprints/1.0 (portfolio project)"}
    for attempt in range(6):  # public Overpass servers are often busy (502/504/429)
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            r = requests.post(url, data={"data": q}, timeout=240, headers=headers)
            r.raise_for_status()
            return r.json()["elements"]
        except (requests.RequestException, ValueError) as err:
            print(f"  {url}: {err}; retrying")
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("Overpass API unavailable; try again later")


def width(tags: dict, kind: str) -> float:
    try:
        return float(str(tags.get("width", "")).replace("m", "").strip())
    except ValueError:
        return DEFAULT_WIDTH_M[kind]


def kind_of(tags: dict) -> str:
    if tags.get("railway") in RAIL:
        return "rail"
    return "foot" if tags.get("highway") in FOOT else "road"


def main():
    elements = query()
    rows = {"bridge_outlines": [], "bridge_ways": [], "railway": [], "platforms": [], "towers": [],
            "power_lines": []}
    for e in elements:
        tags, pts = e.get("tags", {}), [(p["lon"], p["lat"]) for p in e.get("geometry", [])]
        if e["type"] == "node":
            kind = tags.get("power") or tags.get("man_made")
            rows["towers"].append({"osm_id": e["id"], "kind": kind, "height": tags.get("height"),
                                   "tags": json.dumps(tags), "geometry": shapely.Point(e["lon"], e["lat"])})
            continue
        if len(pts) < 2:
            continue
        closed = len(pts) >= 4 and pts[0] == pts[-1]
        line = shapely.LineString(pts)
        base = {"osm_id": e["id"], "name": tags.get("name"), "tags": json.dumps(tags)}
        if tags.get("power") in ("line", "minor_line"):
            rows["power_lines"].append(base | {"power": tags["power"], "geometry": line})
            continue
        if tags.get("man_made") == "bridge" and closed:
            rows["bridge_outlines"].append(base | {"geometry": shapely.Polygon(pts)})
        elif tags.get("railway") == "platform" or tags.get("public_transport") == "platform":
            rows["platforms"].append(base | {"geometry": shapely.Polygon(pts) if closed else line})
        else:
            if tags.get("bridge", "no") != "no":
                k = kind_of(tags)
                rows["bridge_ways"].append(base | {"kind": k, "bridge": tags["bridge"], "width_m": width(tags, k),
                                                   "geometry": line})
            if tags.get("railway") in RAIL and tags.get("tunnel", "no") == "no":
                rows["railway"].append(base | {"railway": tags["railway"], "bridge": tags.get("bridge", "no"),
                                               "geometry": line})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.unlink(missing_ok=True)
    for layer, data in rows.items():
        gdf = gpd.GeoDataFrame(pd.DataFrame(data), geometry="geometry", crs="EPSG:4326").to_crs(config.CRS)
        gdf.to_file(OUT, layer=layer, driver="GPKG")
        extra = f", by kind {gdf['kind'].value_counts().to_dict()}" if layer == "bridge_ways" else ""
        print(f"{layer}: {len(gdf)}{extra}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
