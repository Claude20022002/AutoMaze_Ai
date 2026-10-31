"""Ciel, soleil, ombres et brume (équivalent simplifié des réglages météo de CARLA).

- Dôme de ciel en dégradé (bleu au zénith, blanc laiteux à l'horizon), centré sur la caméra.
- Soleil de fin de matinée avec ombres portées, calculées seulement autour de la voiture (carte d'ombre de
  ``SHADOW_AREA_M`` de côté) : c'est ce qui coûte le moins sur un GPU intégré pour l'effet le plus visible.
- Lumière d'appoint bleutée venant du ciel, ambiante chaude, brume de la couleur de l'horizon.
"""

from __future__ import annotations

import math

import numpy as np

ZENITH = (0.33, 0.56, 0.86)
HORIZON = (0.86, 0.90, 0.94)
SUN_COLOR = (1.0, 0.94, 0.82)
FILL_COLOR = (0.30, 0.36, 0.48)
AMBIENT_COLOR = (0.42, 0.41, 0.40)
SUN_HEADING, SUN_PITCH = -35.0, -48.0         # soleil au sud-est, assez haut
SHADOW_AREA_M = 180.0
SHADOW_MAP_PX = 2048
SKY_RADIUS_M = 1000.0


def sky_dome_mesh(rings: int = 10, segments: int = 24, radius: float = SKY_RADIUS_M):
    """Demi-sphère vue de l'intérieur : positions (N, 3), couleurs (N, 4), triangles (M, 3)."""
    positions, colors, triangles = [], [], []
    for i in range(rings + 1):
        elevation = math.radians(-8 + 98 * i / rings)            # un peu sous l'horizon pour cacher le bord
        t = max(0.0, math.sin(elevation)) ** 0.6
        color = [h + (z - h) * t for h, z in zip(HORIZON, ZENITH)]
        for j in range(segments + 1):
            azimuth = 2 * math.pi * j / segments
            positions.append((radius * math.cos(elevation) * math.cos(azimuth),
                              radius * math.cos(elevation) * math.sin(azimuth), radius * math.sin(elevation)))
            colors.append((*color, 1.0))
    for i in range(rings):
        for j in range(segments):
            a, b = i * (segments + 1) + j, (i + 1) * (segments + 1) + j
            triangles += [(a, b + 1, a + 1), (a, b, b + 1)]       # faces tournées vers l'intérieur
    return np.asarray(positions), np.asarray(colors), np.asarray(triangles, dtype=np.int32)


class Environment:
    def __init__(self, base, shadows: bool = True) -> None:
        from panda3d.core import AmbientLight, CullFaceAttrib, DirectionalLight, Fog, RenderState

        from simulator.render3d.masks import SHADOW_MASK, only_on_main_view
        from simulator.render3d.world_mesh import _geom_node

        self.base = base
        render = base.render
        render.setShaderAuto()                         # éclairage par pixel et ombres
        positions, colors, triangles = sky_dome_mesh()
        normals = -positions / np.linalg.norm(positions, axis=1, keepdims=True)
        self.sky = render.attachNewNode(_geom_node("ciel", positions, normals, colors, triangles))
        self.sky.setBin("background", -10)
        self.sky.setDepthWrite(False)
        self.sky.setLightOff(1)
        self.sky.setFogOff(1)
        self.sky.setShaderOff(1)
        only_on_main_view(self.sky)

        self.sun = render.attachNewNode(DirectionalLight("soleil"))
        self.sun.node().setColor((*SUN_COLOR, 1))
        self.sun.setHpr(SUN_HEADING, SUN_PITCH, 0)
        if shadows:
            self.sun.node().setShadowCaster(True, SHADOW_MAP_PX, SHADOW_MAP_PX)
            self.sun.node().setCameraMask(SHADOW_MASK)
            lens = self.sun.node().getLens()
            lens.setFilmSize(SHADOW_AREA_M, SHADOW_AREA_M)
            lens.setNearFar(10, 900)
            # Faces arrière dans la carte d'ombre : supprime les rayures (« shadow acne ») sur les surfaces éclairées.
            self.sun.node().setInitialState(RenderState.make(CullFaceAttrib.makeReverse()))
        fill = render.attachNewNode(DirectionalLight("ciel_appoint"))
        fill.node().setColor((*FILL_COLOR, 1))
        fill.setHpr(SUN_HEADING + 180, -60, 0)
        ambient = render.attachNewNode(AmbientLight("ambiante"))
        ambient.node().setColor((*AMBIENT_COLOR, 1))
        for light in (self.sun, fill, ambient):
            render.setLight(light)

        fog = Fog("brume")
        fog.setColor(*HORIZON)
        fog.setExpDensity(0.0016)
        render.setFog(fog)
        base.setBackgroundColor(*HORIZON, 1)

    def follow(self, target, camera) -> None:
        """Ciel centré sur la caméra ; carte d'ombre centrée sur la cible (la voiture)."""
        self.sky.setPos(camera.getPos(self.base.render))
        p = target.getPos(self.base.render)
        direction = self.sun.getQuat(self.base.render).getForward()
        self.sun.setPos(p - direction * 400)
