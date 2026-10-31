"""Catalogue des modèles 3D : contenu, licences, traçabilité, et normalisation (avec Panda3D, sans fenêtre)."""

import pytest

from simulator.contracts import CLASSES
from simulator.render3d.build_assets import SOURCES
from simulator.render3d.catalog import ASSETS_3D, ModelEntry, load_catalog, normalization


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def test_obstacle_classes_match_contract(catalog):
    assert tuple(catalog.obstacles) == CLASSES
    assert all(catalog.obstacles[label] for label in CLASSES)


def test_every_model_file_is_committed_and_built_by_build_assets(catalog):
    built = {s.dest for s in SOURCES}
    for entry in catalog.entries():
        assert entry.path.is_file(), entry.model
        assert entry.model in built, f"{entry.model} n'est produit par aucune source de build_assets.py"


def test_every_source_is_used(catalog):
    used = {e.model for e in catalog.entries()} | {"kenney/city-kit-roads/Textures/colormap.png"}
    assert {s.dest for s in SOURCES} == used


def test_credits_list_every_model_url(catalog):
    credits = (ASSETS_3D / "CREDITS.md").read_text(encoding="utf-8")
    for entry in catalog.entries():
        assert entry.url in credits, entry.model
        if entry.license != "CC0":
            assert entry.author in credits


def test_animated_models_have_walk_and_idle(catalog):
    for entry in catalog.obstacles["person"]:
        assert {"walk", "idle"} <= set(entry.animations), entry.model


def test_entry_rejects_bad_size_or_license():
    base = dict(model="x.glb", title="x", author="a", url="u")
    with pytest.raises(ValueError):
        ModelEntry(**base, license="CC0")                              # aucune taille
    with pytest.raises(ValueError):
        ModelEntry(**base, license="CC0", length_m=4.0, height_m=1.5)  # deux tailles
    with pytest.raises(ValueError):
        ModelEntry(**base, license="CC-BY-NC 4.0", length_m=4.0)       # licence refusée


def test_pick_is_deterministic(catalog):
    picks = {catalog.pick("car", [7, 1, 2, 0]).model for _ in range(5)}
    assert len(picks) == 1
    assert catalog.pick("car", [7, 1, 2, 0]) in catalog.obstacles["car"]


def test_normalization_centers_and_grounds_a_box():
    entry = ModelEntry(model="x.glb", title="x", author="a", url="u", license="CC0", length_m=4.0)
    scale, offset = normalization((-1.0, 2.0, 0.5), (1.0, 10.0, 2.5), entry)   # 2 × 8 × 2, avant selon Y
    assert scale == pytest.approx(0.5)
    assert offset == pytest.approx((0.0, -3.0, -0.25))                         # centre en x, y ; base à z = 0


# ---------------------------------------------------------------------- avec Panda3D
@pytest.fixture(scope="module")
def base():
    pytest.importorskip("panda3d")
    pytest.importorskip("gltf")
    from panda3d.core import loadPrcFileData

    loadPrcFileData("", "window-type none\naudio-library-name null")
    from direct.showbase.ShowBase import ShowBase

    b = ShowBase()
    yield b
    b.destroy()


def test_models_load_at_real_size_on_the_ground(base, catalog):
    from simulator.render3d.catalog import load_model

    for entry in catalog.entries():
        node = load_model(entry, base.loader)
        lo, hi = node.getTightBounds()
        size = max(hi.x - lo.x, hi.y - lo.y) if entry.length_m is not None else hi.z - lo.z
        target = entry.length_m if entry.length_m is not None else entry.height_m
        assert size == pytest.approx(target, rel=0.02), entry.model
        assert lo.z == pytest.approx(0.0, abs=0.02), entry.model
        if entry.length_m is not None:   # véhicule tourné vers +Y : plus long que large
            assert hi.y - lo.y >= hi.x - lo.x, entry.model
        node.removeNode()


def test_people_have_their_animations(base, catalog):
    from simulator.render3d.catalog import load_model

    for entry in catalog.obstacles["person"]:
        actor = load_model(entry, base.loader).getPythonTag("actor")
        assert set(entry.animations.values()) <= set(actor.getAnimNames()), entry.model
