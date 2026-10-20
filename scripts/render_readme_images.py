r"""Render the README images to docs/images/ (PyQGIS, no window).

    "C:\Program Files\QGIS 3.44.13\bin\python-qgis-ltr.bat" scripts\render_readme_images.py

Only openly licensed layers are drawn: the LiDAR's own DSM hillshade
(Environment Agency, OGL) under the buildings and tree crowns this project produces
(outputs/canopy/, from 12_canopy_height.py and 13_tree_crowns.py). OS
ground truth isn't redistributable, so it never appears in an image.
"""

import sys
from pathlib import Path

from PIL import Image
from qgis.core import (
    QgsApplication,
    QgsFillSymbol,
    QgsGraduatedSymbolRenderer,
    QgsHillshadeRenderer,
    QgsMapRendererParallelJob,
    QgsMapSettings,
    QgsRasterLayer,
    QgsRectangle,
    QgsRendererRange,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
FOOTPRINTS = ROOT / "outputs" / "canopy" / "buildings.gpkg"
TREES = ROOT / "outputs" / "canopy" / "tree_crowns_filled.gpkg"
TREE_RANGES = [(0, 8, "#c2e699"), (8, 15, "#78c679"), (15, 22, "#31a354"), (22, 100, "#006837")]
# Central Exeter: terraces, the cathedral, the river and the railway
EXTENT = QgsRectangle(291250, 92150, 292650, 93200)


def fill(color: str, outline: str, width: float, alpha: int):
    c = QColor(color)
    return QgsFillSymbol.createSimple({"color": f"{c.red()},{c.green()},{c.blue()},{alpha}",
                                       "outline_color": outline, "outline_width": str(width),
                                       "style": "solid" if alpha else "no"})


def hillshade() -> QgsRasterLayer:
    dsm = QgsRasterLayer(str(ROOT / "data" / "processed" / "dsm.tif"), "DSM hillshade")
    shade = QgsHillshadeRenderer(dsm.dataProvider(), 1, 315, 40)
    shade.setZFactor(1.5)
    dsm.setRenderer(shade)
    return dsm


def render(layers: list, out: Path, size=(1150, 860)) -> None:
    settings = QgsMapSettings()
    settings.setLayers(layers)
    settings.setDestinationCrs(layers[-1].crs())
    settings.setExtent(EXTENT)
    settings.setOutputSize(QSize(*size))
    settings.setBackgroundColor(QColor("white"))
    job = QgsMapRendererParallelJob(settings)
    job.start()
    job.waitForFinished()
    job.renderedImage().save(str(out), "png")
    # 256 colours: about 3x smaller, no visible change
    Image.open(out).convert("RGB").quantize(256, Image.Quantize.MEDIANCUT,
                                            dither=Image.Dither.NONE).save(out, optimize=True)
    print(f"Wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


def draw_all() -> None:
    """All layers live in this function, so they are gone before QGIS shuts down
    (a layer outliving exitQgis crashes Python on exit)."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    shade = hillshade()

    outlines = QgsVectorLayer(str(FOOTPRINTS), "footprints", "ogr")
    outlines.renderer().setSymbol(fill("#00b4ff", "#0060ff", 0.45, 70))
    trees = QgsVectorLayer(str(TREES), "trees", "ogr")
    trees.setRenderer(QgsGraduatedSymbolRenderer("height_max_m", [
        QgsRendererRange(lo, hi, fill(c, "#1b4d2a", 0.1, 170), f"{lo}-{hi} m") for lo, hi, c in TREE_RANGES]))
    render([outlines, trees, shade], IMAGES / "footprints.png")

    heights = QgsVectorLayer(str(FOOTPRINTS), "heights", "ogr")
    ranges = [(0, 5, "#ffffb2", "under 5 m"), (5, 8, "#fecc5c", "5-8 m"), (8, 12, "#fd8d3c", "8-12 m"),
              (12, 20, "#f03b20", "12-20 m"), (20, 200, "#bd0026", "20 m and over")]
    heights.setRenderer(QgsGraduatedSymbolRenderer("height_max_m", [
        QgsRendererRange(lo, hi, fill(c, "#333333", 0.15, 235), label) for lo, hi, c, label in ranges]))
    render([heights, shade], IMAGES / "heights.png")


def main() -> int:
    app = QgsApplication([], False)
    app.initQgis()
    draw_all()
    app.exitQgis()
    return 0


if __name__ == "__main__":
    sys.exit(main())
