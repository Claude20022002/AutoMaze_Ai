"""Bâtiments OpenStreetMap de la zone de la carte, pour la vue 3D.

Téléchargés une seule fois (comme la carte), projetés dans le repère de la carte (UTM, mètres) et enregistrés
dans ``simulator/assets/casablanca_buildings.npz`` : la vue 3D fonctionne ensuite sans connexion internet.
Sans ce fichier, la ville s'affiche sans bâtiments.

Hauteur : balise OSM ``height``, sinon ``building:levels`` × 3,2 m, sinon une hauteur **estimée** (3 à 7 étages,
tirée avec la graine du bâtiment) : ce ne sont pas toujours les vraies hauteurs.

    python -m simulator.render3d.buildings --download

Données © contributeurs OpenStreetMap, licence ODbL.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from simulator.config import MAP_BBOX
from simulator.render3d.world_mesh import building_height

BUILDINGS_FILE = Path(__file__).resolve().parent.parent / "assets" / "casablanca_buildings.npz"


def download(crs, path: Path = BUILDINGS_FILE) -> Path:
    """Télécharge les bâtiments de ``MAP_BBOX`` (miroirs Overpass essayés à la suite) et les enregistre."""
    import osmnx as ox

    from simulator.world.road_map import OVERPASS_SERVERS

    ox.settings.requests_timeout = 300
    last_error: Exception | None = None
    for url in OVERPASS_SERVERS:
        ox.settings.overpass_url = url
        try:
            features = ox.features_from_bbox(MAP_BBOX, tags={"building": True})
            break
        except Exception as error:  # serveur saturé ou indisponible : miroir suivant
            print(f"Échec avec {url} : {type(error).__name__}")
            last_error = error
    else:
        raise RuntimeError("Aucun serveur Overpass n'a répondu, réessayez plus tard.") from last_error
    return save(ox.projection.project_gdf(features, to_crs=crs), path)


def save(features, path: Path = BUILDINGS_FILE) -> Path:
    """Enregistre les contours extérieurs et les hauteurs d'un GeoDataFrame déjà projeté."""
    coords, offsets, heights = [], [0], []
    height_col = features["height"] if "height" in features else None
    levels_col = features["building:levels"] if "building:levels" in features else None
    for i, geom in enumerate(features.geometry):
        polygons = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for polygon in polygons:
            if polygon.geom_type != "Polygon" or polygon.area < 12:
                continue
            ring = np.asarray(polygon.exterior.coords, dtype=np.float64)
            coords.append(ring)
            offsets.append(offsets[-1] + len(ring))
            height = _to_float(height_col.iloc[i]) if height_col is not None else np.nan
            levels = _to_float(levels_col.iloc[i]) if levels_col is not None else np.nan
            heights.append(building_height(height, levels, seed=len(heights)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, coords=np.vstack(coords), offsets=np.asarray(offsets), heights=np.asarray(heights))
    print(f"Bâtiments enregistrés : {path} ({len(heights)} contours)")
    return path


def _to_float(value) -> float:
    try:
        return float(str(value).replace(",", ".").split()[0])
    except (ValueError, IndexError):
        return np.nan


def load(path: Path = BUILDINGS_FILE) -> tuple[list[Polygon], list[float]] | None:
    """(contours en coordonnées de la carte, hauteurs en mètres), ou None si le fichier n'existe pas."""
    if not path.exists():
        return None
    data = np.load(path)
    offsets, coords = data["offsets"], data["coords"]
    polygons = [Polygon(coords[a:b]) for a, b in zip(offsets[:-1], offsets[1:])]
    return polygons, [float(h) for h in data["heights"]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="télécharge les bâtiments depuis OpenStreetMap")
    args = parser.parse_args(argv)
    if not args.download:
        parser.print_help()
        return
    from simulator.world.road_map import load_map

    download(load_map().graph["crs"])


if __name__ == "__main__":
    main()
