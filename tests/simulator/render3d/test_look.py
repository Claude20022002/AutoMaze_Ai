"""Apparence de la ville et interface conducteur : textures, mobilier, zébras, îlots, façades, tableau de bord."""

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from simulator.config import MAP_FILE, SimConfig
from simulator.contracts import Action
from simulator.render3d import street_furniture as sf
from simulator.render3d import textures
from simulator.render3d import world_mesh as wm
from simulator.render3d.app import compute_layout


# ---------------------------------------------------------------------- textures
@pytest.mark.parametrize("style", textures.FACADE_STYLES)
def test_facades_are_light_rgb_tiles_with_dark_windows(style):
    img = textures.facade(style)
    assert img.shape == (textures.SIZE, textures.SIZE, 3) and img.dtype == np.uint8
    assert img.mean() > 140                       # texture claire : la couleur du bâtiment vient des sommets
    assert (img.max(axis=2) < 100).mean() > 0.05  # des vitres sombres
    assert np.array_equal(img, textures.facade(style))


def test_shopfront_variants_differ_and_unknown_style_fails():
    assert not np.array_equal(textures.shopfront(0), textures.shopfront(1))
    with pytest.raises(ValueError):
        textures.facade("gothique")


# ---------------------------------------------------------------------- mobilier urbain
@pytest.mark.parametrize("mesh", [sf.palm(), sf.round_tree(6.0, 1), sf.street_light()])
def test_furniture_meshes_stand_on_the_ground(mesh):
    positions, normals, colors, triangles = mesh
    assert positions[:, 2].min() == pytest.approx(0.0)
    assert positions[:, 2].max() > 5.0
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert triangles.max() < len(positions) and colors.shape == (len(positions), 4)


def test_street_class():
    assert sf.street_class("primary") == "major"
    assert sf.street_class(["secondary_link", "residential"]) == "major"
    assert sf.street_class("tertiary") == "medium"
    assert sf.street_class(None) == "minor"


def test_placements_avoid_roads_and_buildings_and_are_deterministic():
    street = wm.Street(LineString([(0, 0), (200, 0)]), 12.0, True, "major")
    layout = wm.road_layout([street])
    building = Polygon([(60, 7.2), (90, 7.2), (90, 30), (60, 30)])          # bâtiment au bord du trottoir nord
    first = sf.placements([street], layout.roads, [building])
    assert first == sf.placements([street], layout.roads, [building])
    assert {p.kind for p in first} == {"palm", "light"}
    for p in first:
        point = Point(p.x, p.y)
        assert not layout.roads.contains(point)
        assert building.distance(point) > 0.5


# ---------------------------------------------------------------------- chaussée
def _cross_graph(kind: str) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node(0, x=0.0, y=0.0, highway=kind)
    for node, (x, y) in {1: (100.0, 0.0), 2: (-100.0, 0.0), 3: (0.0, 100.0), 4: (0.0, -100.0)}.items():
        graph.add_node(node, x=x, y=y)
        for u, v in ((0, node), (node, 0)):
            graph.add_edge(u, v, 0, length=100.0, highway="secondary", oneway=False)
    return graph


def test_crosswalk_stripes_lie_on_the_road():
    graph = _cross_graph("traffic_signals")
    frame = wm.WorldFrame(0, 0)
    stripes = wm.crosswalks(graph, frame)
    assert len(stripes) == 4 * 10                         # 4 branches de 10 m de large, une bande par mètre
    roads = wm.road_layout(wm.streets(graph, frame)).roads
    assert all(roads.buffer(0.01).contains(stripe) for stripe in stripes)


def test_roundabout_centre_becomes_grass():
    ring = LineString([(30 * np.cos(a), 30 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 41)])
    layout = wm.road_layout([wm.Street(ring, 8.0, False, "major")])
    assert layout.islands.contains(Point(0, 0))
    assert not layout.islands.intersects(layout.roads)


def test_wall_band_counts_whole_bays_and_floors():
    positions, normals, uvs = wm.wall_band(Polygon([(0, 0), (10, 0), (10, 6), (0, 6)]), 4.2, 13.8,
                                           textures.FACADE_BAY_M, textures.FLOOR_M)
    assert positions[:, 2].min() == pytest.approx(4.2) and positions[:, 2].max() == pytest.approx(13.8)
    assert set(np.round(uvs[:, 1], 6)) == {0.0, 3.0}       # 9,6 m = 3 étages entiers
    assert np.allclose(uvs[:, 0], np.round(uvs[:, 0]))     # travées entières : fenêtres jamais coupées


def test_building_style_is_deterministic_and_mostly_white():
    styles = [wm.building_style(i) for i in range(400)]
    assert styles[:5] == [wm.building_style(i) for i in range(5)]
    whites_and_creams = {color for color, _ in wm.BUILDING_COLORS[:3]}
    light = sum(wall in whites_and_creams for _, _, wall, _ in styles)
    assert light / len(styles) > 0.6                      # « ville blanche »


# ---------------------------------------------------------------------- mise en page
def test_layout_keeps_panel_and_dashboard_out_of_the_scene():
    full = compute_layout(1440, 860, 400, driver_view=False)
    assert full.scene == pytest.approx((0, 1040 / 1440, 0, 1)) and full.dashboard_px[3] == 0
    driver = compute_layout(1440, 860, 400, driver_view=True)
    assert driver.scene[2] > 0 and driver.scene_px == (1040, 860 - driver.dashboard_px[3])


# ---------------------------------------------------------------------- tableau de bord
needs_map = pytest.mark.skipif(not MAP_FILE.exists(), reason="carte absente")


@pytest.fixture(scope="module")
def episode_factory():
    from simulator.engine import Episode
    from simulator.main import load_perception
    from simulator.world.road_map import load_map

    graph, config = load_map(), SimConfig()
    classifier, images = load_perception("simulation", config)
    return lambda: Episode(graph, classifier, config, 0, image_for=images)


@needs_map
def test_dashboard_reads_the_episode(episode_factory):
    from simulator.render3d.dashboard import driving_state, object_ahead, object_text, remaining_m

    episode = episode_factory()
    assert driving_state(episode).text == "Voie libre"
    start = remaining_m(episode)
    assert start == pytest.approx(episode.planned.length_m)
    for _ in range(10):
        episode.step()
    assert remaining_m(episode) < start
    ahead = object_ahead(episode)
    while ahead is None:                                  # premier objet qui entre dans les 80 m
        episode.step()
        ahead = object_ahead(episode)
    assert ahead.label is None and ahead.distance_m <= 80  # pas encore classé : la voiture le voit sans le connaître
    assert object_text(ahead) == f"Objet à {ahead.distance_m:.0f} m · non classé"
    assert episode.last_detection is None


@needs_map
def test_dashboard_names_the_stop_reason(episode_factory):
    from simulator.render3d.dashboard import driving_state

    episode = episode_factory()
    while episode.last_detection is None or episode.last_detection.action != Action.STOP_WAIT:
        episode.step()
        assert not episode.done, "aucun piéton dans l'épisode 0"
    if episode.wait > 0:
        assert driving_state(episode).text.startswith("Arrêt : piéton")
