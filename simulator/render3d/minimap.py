"""Minicarte : la même scène vue de dessus par une seconde caméra orthographique, orientée dans le sens de marche.

Masques de caméra (``masks.py``) : les marqueurs (voiture, obstacles) ne sont visibles que sur la minicarte ;
les étiquettes 3D ne le sont que dans la vue principale.
"""

from __future__ import annotations

import numpy as np

from simulator.render3d.masks import MAIN_MASK, MINIMAP_MASK, only_on_main_view, only_on_minimap  # noqa: F401

VIEW_M = 260.0          # côté de la zone affichée, en mètres
MARKER_Z = 160.0        # au-dessus des plus hauts bâtiments


class MiniMap:
    def __init__(self, base, ego_node) -> None:
        from panda3d.core import Camera, FogAttrib, OrthographicLens, RenderState

        self.base, self.ego = base, ego_node
        base.camNode.setCameraMask(MAIN_MASK)
        lens = OrthographicLens()
        lens.setFilmSize(VIEW_M, VIEW_M)
        lens.setNearFar(1, 1000)
        node = Camera("minicarte", lens)
        node.setCameraMask(MINIMAP_MASK)
        node.setInitialState(RenderState.make(FogAttrib.makeOff()))   # pas de brume vue d'en haut
        self.camera = base.render.attachNewNode(node)
        self.region = base.win.makeDisplayRegion(0, 0.1, 0, 0.1)
        self.region.setCamera(self.camera)
        self.region.setSort(20)
        self.region.setClearColorActive(True)
        self.region.setClearColor((0.86, 0.82, 0.73, 1))
        self.region.setClearDepthActive(True)
        self.marker = self._arrow()
        self.marker.reparentTo(ego_node)
        self.marker.setPos(0, 0, MARKER_Z)

    def _arrow(self):
        from panda3d.core import NodePath

        from simulator.render3d.world_mesh import _geom_node

        pos = np.array([(0, 6, 0), (-4, -5, 0), (0, -2, 0), (0, 6, 0), (0, -2, 0), (4, -5, 0)], dtype=float) * 1.8
        nrm = np.tile([0.0, 0.0, 1.0], (6, 1))
        col = np.tile([0.10, 0.45, 0.95, 1.0], (6, 1))
        node = NodePath(_geom_node("fleche", pos, nrm, col, np.arange(6).reshape(-1, 3)))
        node.setTwoSided(True)
        node.setLightOff(1)
        node.setDepthTest(False)
        node.setBin("fixed", 50)
        only_on_minimap(node)
        return node

    def obstacle_marker(self, parent, color):
        """Disque coloré au-dessus d'un obstacle, visible seulement sur la minicarte."""
        from panda3d.core import CardMaker

        card = CardMaker("repere")
        card.setFrame(-3.5, 3.5, -3.5, 3.5)
        marker = parent.attachNewNode(card.generate())
        marker.setP(-90)
        marker.setZ(MARKER_Z)
        marker.setColor(*color, 1)
        marker.setLightOff(1)
        marker.setDepthTest(False)
        marker.setBin("fixed", 40)
        only_on_minimap(marker)
        return marker

    def set_region(self, left: float, right: float, bottom: float, top: float) -> None:
        self.region.setDimensions(left, right, bottom, top)

    def update(self) -> None:
        p = self.ego.getPos(self.base.render)
        self.camera.setPos(p.x, p.y, 500)
        self.camera.setHpr(self.ego.getH(self.base.render), -90, 0)   # le haut de la carte = sens de marche
