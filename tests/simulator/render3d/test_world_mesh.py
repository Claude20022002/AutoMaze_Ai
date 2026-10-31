"""Géométrie de la ville 3D (sans GPU) et vue 3D complète (capture hors écran, dans un processus séparé)."""

import math
import subprocess
import sys

import pytest
from shapely.geometry import LineString, Point, Polygon

from simulator.config import MAP_FILE, SimConfig
from simulator.render3d import world_mesh as wm


def test_frame_round_trip():
    frame = wm.WorldFrame(330_000.0, 3_716_000.0)
    assert frame.to_world(*frame.to_local(331_234.5, 3_715_321.25)) == pytest.approx((331_234.5, 3_715_321.25))


def test_heading_points_model_front_along_direction():
    assert wm.heading_deg(0, 1) == pytest.approx(0)       # +Y : cap nul
    assert wm.heading_deg(-1, 0) == pytest.approx(90)     # Panda3D : sens trigonométrique
    assert wm.heading_deg(1, 0) == pytest.approx(-90)


@pytest.mark.parametrize("data, width", [
    ({"width": "9"}, 9.0),
    ({"width": ["7 m", "8"]}, 7.0),
    ({"lanes": "2"}, 6.4),
    ({"lanes": ["3", "2"], "highway": "primary"}, 9.6),
    ({"highway": "primary"}, 12.0),
    ({"highway": ["residential", "service"]}, 6.5),
    ({"highway": "secondary_link"}, 10.0),
    ({"width": "n/a", "highway": "inconnu"}, wm.DEFAULT_WIDTH_M),
])
def test_road_width(data, width):
    assert wm.road_width(data) == pytest.approx(width)


def test_triangulation_keeps_area_and_faces_up():
    polygon = Polygon([(0, 0), (4, 0), (4, 3), (2, 1), (0, 3)], holes=[[(0.5, 0.3), (1, 0.3), (1, 0.6)]])
    points, triangles = wm.triangulate(polygon)
    area = 0.0
    for a, b, c in triangles:
        (x0, y0), (x1, y1), (x2, y2) = points[a], points[b], points[c]
        cross = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        assert cross > 0                                   # sens trigonométrique : face vers +Z
        area += cross / 2
    assert area == pytest.approx(polygon.area)


def test_extrude_walls_face_outward_and_roof_on_top():
    positions, normals, triangles = wm.extrude(Polygon([(0, 0), (10, 0), (10, 6), (0, 6)]), 9.0)
    assert positions[:, 2].max() == pytest.approx(9.0)
    centre = positions[:, :2].mean(axis=0)
    for p, n in zip(positions, normals):
        if n[2] == 0:                                      # mur : la normale s'éloigne du centre
            assert (p[0] - centre[0]) * n[0] + (p[1] - centre[1]) * n[1] > 0


def test_building_height_prefers_osm_tags():
    assert wm.building_height(25.0, float("nan"), seed=1) == 25.0
    assert wm.building_height(float("nan"), 4, seed=1) == pytest.approx(4 * wm.LEVEL_HEIGHT_M)
    estimated = wm.building_height(float("nan"), float("nan"), seed=1)
    assert estimated == wm.building_height(float("nan"), float("nan"), seed=1)   # même graine, même hauteur
    assert 3 * wm.LEVEL_HEIGHT_M <= estimated <= 7 * wm.LEVEL_HEIGHT_M


def test_two_way_street_gives_one_ribbon_with_markings():
    import networkx as nx

    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=100.0, y=0.0)
    graph.add_edge(1, 2, 0, length=100.0, highway="secondary", oneway=False)
    graph.add_edge(2, 1, 0, length=100.0, highway="secondary", oneway=False)
    streets = wm.streets(graph, wm.WorldFrame(0, 0))
    assert len(streets) == 1 and streets[0].two_way
    layout = wm.road_layout(streets)
    assert layout.roads.area == pytest.approx(100 * 10 + math.pi * 5 ** 2, rel=0.02)   # ruban + bouts arrondis
    assert not layout.markings.is_empty
    assert not layout.sidewalks.intersects(Point(50, 0))


# ---------------------------------------------------------------------- sur la vraie carte
needs_map = pytest.mark.skipif(not MAP_FILE.exists(), reason="carte absente")


@pytest.fixture(scope="module")
def city():
    from simulator.world.road_map import load_map

    graph = load_map()
    frame = wm.WorldFrame.from_graph(graph)
    return graph, frame, wm.road_layout(wm.streets(graph, frame))


@needs_map
def test_local_coordinates_stay_small(city):
    graph, frame, layout = city
    minx, miny, maxx, maxy = layout.roads.bounds
    assert max(abs(minx), abs(miny), abs(maxx), abs(maxy)) < 5_000


@needs_map
def test_default_route_and_lane_points_are_on_the_road(city):
    from simulator.render3d.actors import edge_pose, lane_point
    from simulator.world.pathfinding import dijkstra
    from simulator.world.road_map import landmark_node

    graph, frame, layout = city
    config = SimConfig()
    route = dijkstra(graph, landmark_node(graph, config.start), landmark_node(graph, config.goal))
    road = layout.roads.buffer(0.3)
    for edge in route.edges:
        x, y, dx, dy = edge_pose(graph, edge, float(graph.edges[edge]["length"]) / 2)
        assert road.contains(Point(frame.to_local(x, y))), edge
        assert road.contains(Point(lane_point(frame, x, y, dx, dy, wm.LANE_OFFSET_M))), edge


@needs_map
def test_props_stand_off_the_carriageway(city):
    graph, frame, layout = city
    placements = wm.prop_placements(graph, frame, layout.roads)
    assert {p.kind for p in placements} <= {"traffic_signals", "stop", "crossing"}
    assert placements
    assert not [p for p in placements if layout.roads.contains(Point(p.x, p.y))]


@needs_map
def test_3d_view_screenshot(tmp_path):
    pytest.importorskip("panda3d")
    out = tmp_path / "vue3d.png"
    done = subprocess.run([sys.executable, "-m", "simulator.render3d.app", "--screenshot", str(out),
                           "--steps", "30", "--size", "640x400"], capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]
    from PIL import Image

    image = Image.open(out)
    assert image.size == (640, 400)
    assert len(image.convert("RGB").getcolors(maxcolors=1 << 20)) > 200  # une vraie scène, pas un écran uni
