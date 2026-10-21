"""Road network for the viewer's route tool, from OS Open Roads.

OS Open Roads road links carry their start and end junctions (startNode, endNode), so
they form a routing graph as they are. Links touching the AOI plus MARGIN_M are kept, so
a route between two points in the AOI can leave it, as a real one would.

Each link gets a simple class for drawing, its length, and the modes allowed on it:
no walking or cycling on motorways, no driving on restricted local access roads
(private and gated roads) or shared-use carriageways (pedestrianised streets). Driving
speeds are typical free-flow speeds by road function and form (CAR_KMH); the data has
no one-way streets, turn restrictions or traffic, so driving times are an estimate. Walking and cycling use WALK_KMH and CYCLE_KMH on every
road they may use; footpaths are not in OS Open Roads.

Writes outputs/roads/road_network.gpkg (EPSG:27700), junctions renumbered 0..n-1;
main_network marks the largest connected part (the rest are fragments cut by the margin).

    python 15_road_network.py
"""

import geopandas as gpd
import numpy as np
import pandas as pd

import config

OUT = config.OUTPUTS / "roads" / "road_network.gpkg"
MARGIN_M = 1000
WALK_KMH, CYCLE_KMH = 4.8, 15.0
# OS function -> class drawn in the viewer
CLASS = {"Motorway": "motorway", "A Road": "a_road", "B Road": "b_road", "Minor Road": "minor",
         "Local Road": "local", "Local Access Road": "access", "Secondary Access Road": "access",
         "Restricted Local Access Road": "restricted"}
CAR_KMH = {"Motorway": 100, "A Road": 48, "B Road": 40, "Minor Road": 35, "Local Road": 30,
           "Local Access Road": 20, "Secondary Access Road": 15, "Restricted Local Access Road": 0}
DUAL_A_KMH, SLIP_KMH, ROUNDABOUT_KMH = 80, 50, 25


def car_speed(row) -> float:
    # restricted access (private, gated) and shared-use carriageways (pedestrianised streets)
    if row["function"] == "Restricted Local Access Road" or row["formOfWay"] == "Shared Use Carriageway":
        return 0.0
    if row["formOfWay"] == "Roundabout":
        return ROUNDABOUT_KMH
    if row["formOfWay"] == "Slip Road":
        return SLIP_KMH
    if row["function"] == "A Road" and "Dual" in row["formOfWay"]:
        return DUAL_A_KMH
    return CAR_KMH[row["function"]]


def main_part(a, b, n):
    """True for junctions in the largest connected part of the graph (union-find)."""
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in zip(a, b):
        parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n)])
    return roots == np.bincount(roots).argmax()


def main() -> None:
    minx, miny, maxx, maxy = config.ANALYSIS_BOUNDS
    bounds = (minx - MARGIN_M, miny - MARGIN_M, maxx + MARGIN_M, maxy + MARGIN_M)
    r = gpd.read_file(config.ROADS_PATH, bbox=bounds)
    r = r[r["fictitious"] != "true"]
    r["geometry"] = r.geometry.force_2d()

    junctions = pd.Index(pd.unique(pd.concat([r["startNode"], r["endNode"]])))
    out = gpd.GeoDataFrame({
        "from_node": junctions.get_indexer(r["startNode"]),
        "to_node": junctions.get_indexer(r["endNode"]),
        "road_class": r["function"].map(CLASS),
        "name": r["name1"].fillna(""),
        "number": r["roadNumber"].fillna(""),
        "form": r["formOfWay"],
        "length_m": r.length.round(1),
        "car_kmh": r.apply(car_speed, axis=1),
        "walk": r["function"] != "Motorway",
        "cycle": r["function"] != "Motorway",
    }, geometry=r.geometry, crs=r.crs)
    assert out["road_class"].notna().all(), "a road function missing from CLASS"
    in_main = main_part(out["from_node"], out["to_node"], len(junctions))
    out["main_network"] = in_main[out["from_node"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.unlink(missing_ok=True)
    out.to_file(OUT, layer="road_network", driver="GPKG")
    km = out.groupby("road_class")["length_m"].sum().div(1000).round(1).sort_values(ascending=False)
    print(f"{len(out):,} links, {len(junctions):,} junctions, {out['length_m'].sum() / 1000:,.0f} km "
          f"(AOI + {MARGIN_M} m); km by class {km.to_dict()} -> {OUT}")
    print(f"{(~out['main_network']).sum()} links in {MARGIN_M} m-margin fragments left out of routing")


if __name__ == "__main__":
    main()
