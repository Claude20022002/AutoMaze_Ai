"""Planche de contrôle des modèles 3D : tous les modèles du catalogue à leur taille réelle, sur une grille de 1 m.

Sert à vérifier après chaque ajout : échelle, orientation (flèche rouge = avant attendu, vers +Y),
modèle posé au sol, textures présentes. Rendu hors écran, sans fenêtre.

    python -m simulator.render3d.contact_sheet                 # reports/simulator/planche_assets.png
    python -m simulator.render3d.contact_sheet --view top      # vue de dessus
    python -m simulator.render3d.contact_sheet --view fronts   # chaque modèle vu depuis son avant attendu (+Y)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from simulator.config import REPORTS_DIR
from simulator.render3d.catalog import Catalog, ModelEntry, load_catalog, load_model

WIDTH, HEIGHT = 1800, 1000
TILE = 300          # côté d'une vignette de la vue « fronts », en pixels
TILE_COLUMNS = 6
ROW_SPACING_M = 16.0
GAP_M = 1.5


def _rows(catalog: Catalog) -> list[list[tuple[str, ModelEntry]]]:
    """Une rangée pour les petits obstacles, une pour les gros véhicules, une pour l'ego et la signalisation."""
    small = [(f"{label} {i + 1}", e) for label in ("person", "bicycle", "motorcycle")
             for i, e in enumerate(catalog.obstacles[label])]
    large = [(f"{label} {i + 1}", e) for label in ("car", "bus", "truck")
             for i, e in enumerate(catalog.obstacles[label])]
    other = [("ego", catalog.ego), *catalog.props.items()]
    return [small, large, other]


def _footprint(entry: ModelEntry) -> float:
    return entry.length_m if entry.length_m is not None else 1.0


def render(catalog: Catalog, view: str = "perspective") -> np.ndarray:
    """Image RGB (HEIGHT, WIDTH, 3) de la planche."""
    from panda3d.core import (AmbientLight, CardMaker, DirectionalLight, LineSegs, TextNode, loadPrcFileData)

    loadPrcFileData("", f"window-type offscreen\nwin-size {WIDTH} {HEIGHT}\naudio-library-name null\nsync-video false")
    from direct.showbase.ShowBase import ShowBase

    base = ShowBase()
    try:
        base.setBackgroundColor(0.93, 0.93, 0.9, 1)
        sun = base.render.attachNewNode(DirectionalLight("soleil"))
        sun.node().setColor((0.9, 0.9, 0.85, 1))
        sun.setHpr(-30, -55, 0)
        ambient = base.render.attachNewNode(AmbientLight("ambiante"))
        ambient.node().setColor((0.45, 0.45, 0.5, 1))
        base.render.setLight(sun)
        base.render.setLight(ambient)

        rows = _rows(catalog)
        widths = [sum(_footprint(e) + GAP_M for _, e in row) for row in rows]
        extent_x = max(widths)
        extent_y = ROW_SPACING_M * (len(rows) - 1)

        # Sol et grille de 1 m
        ground = CardMaker("sol")
        ground.setFrame(-2, extent_x + 2, -6, extent_y + 8)
        floor = base.render.attachNewNode(ground.generate())
        floor.setP(-90)
        floor.setColor(0.82, 0.82, 0.78, 1)
        grid = LineSegs("grille")
        grid.setColor(0.65, 0.65, 0.6, 1)
        for x in range(-2, int(extent_x) + 3):
            grid.moveTo(x, -6, 0.01)
            grid.drawTo(x, extent_y + 8, 0.01)
        for y in range(-6, int(extent_y) + 9):
            grid.moveTo(-2, y, 0.01)
            grid.drawTo(extent_x + 2, y, 0.01)
        base.render.attachNewNode(grid.create())

        for r, row in enumerate(rows):
            x = 0.0
            for name, entry in row:
                half = _footprint(entry) / 2
                node = load_model(entry, base.loader)
                node.reparentTo(base.render)
                node.setPos(x + half, r * ROW_SPACING_M, 0)
                actor = node.getPythonTag("actor")
                if actor is not None:
                    actor.pose(entry.animations["idle"], 0)
                # Flèche rouge au sol : direction +Y, où l'avant du modèle doit pointer
                arrow = LineSegs("avant")
                arrow.setThickness(3)
                arrow.setColor(0.85, 0.1, 0.1, 1)
                tip = max(1.0, half + 1.0)
                for a, b in (((0, -0.5), (0, tip)), ((-0.4, tip - 0.5), (0, tip)), ((0.4, tip - 0.5), (0, tip))):
                    arrow.moveTo(x + half + a[0], r * ROW_SPACING_M + a[1], 0.03)
                    arrow.drawTo(x + half + b[0], r * ROW_SPACING_M + b[1], 0.03)
                base.render.attachNewNode(arrow.create())
                label = TextNode(name)
                label.setText(name)
                label.setAlign(TextNode.ACenter)
                label.setTextColor(0.1, 0.1, 0.1, 1)
                text = base.render.attachNewNode(label)
                text.setScale(0.55)
                text.setPos(x + half, r * ROW_SPACING_M - 2.6, 0.05)
                text.setP(-90 if view == "top" else -30)
                x += 2 * half + GAP_M

        lens = base.camLens
        lens.setFov(40)
        center = (extent_x / 2, extent_y / 2)
        if view == "top":
            base.camera.setPos(center[0], center[1], max(extent_x, extent_y) * 1.45)
            base.camera.lookAt(center[0], center[1] + 0.001, 0)
        else:
            base.camera.setPos(center[0] - 4, -34, 30)
            base.camera.lookAt(center[0], center[1] - 1, 0)

        return _capture(base)
    finally:
        base.destroy()


def _capture(base) -> np.ndarray:
    base.graphicsEngine.renderFrame()
    base.graphicsEngine.renderFrame()
    tex = base.win.getScreenshot()
    image = np.frombuffer(tex.getRamImageAs("RGB"), np.uint8).reshape(tex.getYSize(), tex.getXSize(), 3)
    return np.ascontiguousarray(image[::-1])     # OpenGL : origine en bas


def render_fronts(catalog: Catalog) -> np.ndarray:
    """Une vignette par modèle, vu de trois quarts avant : un modèle bien orienté montre sa face avant."""
    from panda3d.core import AmbientLight, DirectionalLight, loadPrcFileData
    from PIL import Image, ImageDraw

    loadPrcFileData("", f"window-type offscreen\nwin-size {TILE} {TILE}\naudio-library-name null\nsync-video false")
    from direct.showbase.ShowBase import ShowBase

    base = ShowBase()
    try:
        base.setBackgroundColor(0.93, 0.93, 0.9, 1)
        sun = base.render.attachNewNode(DirectionalLight("soleil"))
        sun.setHpr(150, -40, 0)                   # éclaire l'avant (+Y) du modèle
        ambient = base.render.attachNewNode(AmbientLight("ambiante"))
        ambient.node().setColor((0.45, 0.45, 0.5, 1))
        base.render.setLight(sun)
        base.render.setLight(ambient)
        base.camLens.setFov(35)
        tiles = []
        for name, entry in [t for row in _rows(catalog) for t in row]:
            node = load_model(entry, base.loader)
            node.reparentTo(base.render)
            lo, hi = node.getTightBounds()
            size = max(hi - lo)
            center = (lo + hi) / 2
            base.camera.setPos(center.x + 0.9 * size, center.y + 1.6 * size, center.z + 0.7 * size)
            base.camera.lookAt(center)
            tile = Image.fromarray(_capture(base))
            ImageDraw.Draw(tile).text((8, 6), name, fill=(20, 20, 20))
            tiles.append(tile)
            node.removeNode()
        rows = -(-len(tiles) // TILE_COLUMNS)
        sheet = Image.new("RGB", (TILE * TILE_COLUMNS, TILE * rows), (255, 255, 255))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, (TILE * (i % TILE_COLUMNS), TILE * (i // TILE_COLUMNS)))
        return np.asarray(sheet)
    finally:
        base.destroy()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", choices=("perspective", "top", "fronts"), default="perspective")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    from PIL import Image

    names = {"perspective": "planche_assets.png", "top": "planche_assets_dessus.png", "fronts": "planche_assets_avants.png"}
    out = args.out or REPORTS_DIR / names[args.view]
    out.parent.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    image = render_fronts(catalog) if args.view == "fronts" else render(catalog, args.view)
    Image.fromarray(image).save(out)
    print(f"Planche enregistrée : {out}")


if __name__ == "__main__":
    main()
