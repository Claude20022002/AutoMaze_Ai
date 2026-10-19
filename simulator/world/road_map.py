"""Carte routière réelle d'une partie de Casablanca, issue d'OpenStreetMap via OSMnx.

La carte est téléchargée une seule fois puis enregistrée dans ``simulator/assets/`` :
le simulateur fonctionne ensuite sans connexion internet.

Données : © contributeurs OpenStreetMap, licence ODbL (https://www.openstreetmap.org/copyright).

Commandes :
    python -m simulator.world.road_map --download            # (re)télécharge la zone de config.MAP_BBOX
    python -m simulator.world.road_map --preview             # image de la carte et de l'itinéraire par défaut
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

from simulator.config import LANDMARKS, MAP_BBOX, MAP_FILE, REPORTS_DIR, SimConfig

OVERPASS_SERVERS = (  # le serveur principal est souvent saturé : on essaie les miroirs à la suite
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
)


def download_map(path: Path = MAP_FILE, bbox: tuple[float, float, float, float] = MAP_BBOX) -> Path:
    """Télécharge le réseau routier carrossable de la zone et l'enregistre en GraphML."""
    import osmnx as ox

    last_error: Exception | None = None
    for url in OVERPASS_SERVERS:
        ox.settings.overpass_url = url
        try:
            graph = ox.graph_from_bbox(bbox=bbox, network_type="drive")
            break
        except Exception as error:  # serveur saturé ou indisponible : miroir suivant
            print(f"Échec avec {url} : {type(error).__name__}")
            last_error = error
    else:
        raise RuntimeError("Aucun serveur Overpass n'a répondu, réessayez plus tard.") from last_error

    # Garde la plus grande composante fortement connexe : depuis chaque intersection, on peut rejoindre
    # toutes les autres en respectant les sens uniques (sinon certains itinéraires seraient impossibles).
    graph = ox.truncate.largest_component(graph, strongly=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, path)
    print(f"Carte enregistrée : {path} ({len(graph)} intersections, {graph.number_of_edges()} tronçons)")
    return path


def load_map(path: Path = MAP_FILE) -> nx.MultiDiGraph:
    """Charge la carte et la projette en mètres (UTM) : x et y deviennent des coordonnées en mètres."""
    import osmnx as ox

    if not path.exists():
        raise FileNotFoundError(f"Carte absente : {path}. Lancez : python -m simulator.world.road_map --download")
    return ox.project_graph(ox.load_graphml(path))


def nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Intersection la plus proche d'un point GPS (latitude, longitude)."""
    import osmnx as ox
    from pyproj import Transformer

    to_graph_crs = Transformer.from_crs("EPSG:4326", graph.graph["crs"], always_xy=True)
    x, y = to_graph_crs.transform(lon, lat)
    return int(ox.distance.nearest_nodes(graph, x, y))


def landmark_node(graph: nx.MultiDiGraph, name: str) -> int:
    """Intersection la plus proche d'un lieu de ``config.LANDMARKS``."""
    if name not in LANDMARKS:
        raise KeyError(f"Lieu inconnu : {name!r}. Lieux disponibles : {', '.join(LANDMARKS)}")
    return nearest_node(graph, *LANDMARKS[name])


def street_name(graph: nx.MultiDiGraph, edge: tuple[int, int, int]) -> str:
    """Nom de la rue d'un tronçon (OSM peut en donner plusieurs, ou aucun)."""
    name = graph.edges[edge].get("name")
    if isinstance(name, list):
        return " / ".join(name)
    return name or "Rue sans nom"


def save_preview(graph: nx.MultiDiGraph, config: SimConfig, out: Path) -> Path:
    """Image de contrôle : la carte et l'itinéraire le plus court entre le départ et l'arrivée."""
    import matplotlib

    matplotlib.use("Agg")
    import osmnx as ox

    from simulator.world.pathfinding import dijkstra

    route = dijkstra(graph, landmark_node(graph, config.start), landmark_node(graph, config.goal))
    fig, ax = ox.plot_graph_route(
        graph, list(route.nodes), route_color="tab:blue", route_linewidth=4, node_size=0,
        bgcolor="white", edge_color="#bbbbbb", show=False, close=False,
    )
    ax.set_title(f"{config.start} → {config.goal} : {route.length_m / 1000:.2f} km", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="télécharge la carte depuis OpenStreetMap")
    parser.add_argument("--preview", action="store_true", help="enregistre une image de l'itinéraire par défaut")
    args = parser.parse_args()
    if args.download:
        download_map()
    if args.preview:
        out = save_preview(load_map(), SimConfig(), REPORTS_DIR / "apercu_itineraire.png")
        print(f"Aperçu enregistré : {out}")
    if not (args.download or args.preview):
        parser.print_help()


if __name__ == "__main__":
    main()
