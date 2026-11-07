"""Catalogue des modèles 3D (équivalent de la « blueprint library » de CARLA).

``simulator/assets/3d/catalog.json`` associe chaque classe d'obstacle, la voiture de l'ego et les éléments de
signalisation à des modèles .glb, avec leur taille réelle, leur cap et leur licence. Les fichiers sources ne sont
jamais modifiés : l'échelle, l'orientation et l'origine sont corrigées au chargement (``load_model``).

La lecture et la validation du catalogue n'importent pas Panda3D ; seul le chargement des modèles en a besoin.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from simulator.contracts import CLASSES

if TYPE_CHECKING:
    from panda3d.core import NodePath

ASSETS_3D = Path(__file__).resolve().parent.parent / "assets" / "3d"
CATALOG_FILE = ASSETS_3D / "catalog.json"
LICENSES = frozenset({"CC0", "CC-BY 3.0", "CC-BY 4.0"})  # licences acceptées (skill asset-pipeline)


@dataclass(frozen=True)
class ModelEntry:
    """Un modèle du catalogue. Exactement une taille cible : ``length_m`` (véhicules) ou ``height_m``."""

    model: str                      # chemin relatif à simulator/assets/3d/
    title: str
    author: str
    license: str
    url: str
    length_m: float | None = None   # plus grande dimension horizontale (véhicules)
    height_m: float | None = None   # hauteur (personnes, panneaux, feux)
    heading: float = 0.0            # rotation (degrés) qui tourne l'avant du modèle vers +Y
    animations: dict[str, str] = field(default_factory=dict)  # rôle ("walk", "idle") -> nom dans le fichier

    def __post_init__(self) -> None:
        if (self.length_m is None) == (self.height_m is None):
            raise ValueError(f"{self.model} : il faut exactement une taille, length_m ou height_m")
        size = self.length_m if self.length_m is not None else self.height_m
        if size <= 0:
            raise ValueError(f"{self.model} : taille strictement positive attendue")
        if self.license not in LICENSES:
            raise ValueError(f"{self.model} : licence {self.license!r} non acceptée (attendu : {sorted(LICENSES)})")

    @property
    def path(self) -> Path:
        return ASSETS_3D / self.model

    @property
    def animated(self) -> bool:
        return bool(self.animations)


@dataclass(frozen=True)
class Catalog:
    obstacles: dict[str, tuple[ModelEntry, ...]]
    ego: ModelEntry
    props: dict[str, ModelEntry]

    def entries(self) -> Iterator[ModelEntry]:
        for variants in self.obstacles.values():
            yield from variants
        yield self.ego
        yield from self.props.values()

    def pick(self, label: str, seed: int | Sequence[int]) -> ModelEntry:
        """Modèle d'un obstacle, tiré à partir de sa graine : même obstacle = même modèle pour tous les classifieurs."""
        variants = self.obstacles[label]
        return variants[int(np.random.default_rng(seed).integers(len(variants)))]


def load_catalog(path: Path = CATALOG_FILE) -> Catalog:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    obstacles = {label: tuple(ModelEntry(**e) for e in entries) for label, entries in data["obstacles"].items()}
    if tuple(obstacles) != CLASSES:
        raise ValueError(f"classes du catalogue {tuple(obstacles)} différentes de {CLASSES} (même ordre attendu)")
    if not all(obstacles.values()):
        raise ValueError("chaque classe doit avoir au moins un modèle")
    return Catalog(
        obstacles=obstacles,
        ego=ModelEntry(**data["ego"]),
        props={name: ModelEntry(**e) for name, e in data["props"].items()},
    )


def normalization(lo: Sequence[float], hi: Sequence[float], entry: ModelEntry) -> tuple[float, tuple[float, float, float]]:
    """Échelle et translation qui donnent au modèle sa taille réelle, centré en x et y, posé sur le sol (z = 0).

    ``lo`` et ``hi`` : coins de la boîte englobante du modèle déjà tourné de ``entry.heading``.
    """
    dx, dy, dz = (h - l for l, h in zip(lo, hi))
    measured = max(dx, dy) if entry.length_m is not None else dz
    if measured <= 0:
        raise ValueError(f"{entry.model} : boîte englobante vide")
    scale = (entry.length_m if entry.length_m is not None else entry.height_m) / measured
    offset = (-(lo[0] + hi[0]) / 2 * scale, -(lo[1] + hi[1]) / 2 * scale, -lo[2] * scale)
    return scale, offset


def load_model(entry: ModelEntry, loader=None) -> NodePath:
    """Charge et normalise un modèle. Les modèles animés sont des ``Actor`` (``root.find("**/+Character")``)."""
    from panda3d.core import Filename, NodePath

    root = NodePath(entry.title)
    if entry.animated:
        from direct.actor.Actor import Actor

        model = Actor(Filename.fromOsSpecific(str(entry.path)))
    else:
        if loader is None:
            raise ValueError("un loader Panda3D (base.loader) est nécessaire pour un modèle statique")
        model = loader.loadModel(Filename.fromOsSpecific(str(entry.path)))
    model.reparentTo(root)
    model.setH(entry.heading)
    if entry.animated:
        # Les bornes d'un personnage au repos (T-pose, peau en unités du squelette) ne sont pas fiables :
        # on mesure dans la pose d'attente, celle qui est affichée par défaut.
        model.pose(entry.animations["idle"], 0)
        model.update(force=True)
    lo, hi = model.getTightBounds(root)
    scale, offset = normalization(tuple(lo), tuple(hi), entry)
    model.setScale(scale)
    model.setPos(*offset)
    root.setPythonTag("actor", model if entry.animated else None)
    return root
