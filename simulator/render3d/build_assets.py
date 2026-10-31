"""Prépare les modèles 3D commités dans ``simulator/assets/3d/`` à partir des téléchargements bruts.

Les téléchargements (zips extraits dans ``simulator/assets/3d/_downloads/``) ne sont pas dans Git. Ce script
dit d'où vient chaque modèle commité : copie d'un .glb, ou conversion OBJ -> .glb avec Blender.
Sources, auteurs et licences : ``simulator/assets/3d/CREDITS.md``.

    python -m simulator.render3d.build_assets            # (re)construit tout
    python -m simulator.render3d.build_assets --check    # liste les sources manquantes, sans rien écrire

Blender : variable d'environnement BLENDER, sinon ``blender`` dans le PATH, sinon l'emplacement par défaut de Blender 5.2.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ASSETS_3D = Path(__file__).resolve().parent.parent / "assets" / "3d"
DOWNLOADS = ASSETS_3D / "_downloads"
BLENDER_SCRIPT = Path(__file__).resolve().parent / "blender_to_glb.py"
BLENDER_DEFAULT = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")


@dataclass(frozen=True)
class AssetSource:
    dest: str               # chemin relatif à simulator/assets/3d/
    src: str                # chemin relatif à simulator/assets/3d/_downloads/
    reexport: bool = False  # .glb à repasser par Blender (structure refusée par panda3d-gltf)
    colors: dict[str, tuple[float, float, float]] | None = field(default=None, compare=False)  # matériau -> RGB

    @property
    def needs_blender(self) -> bool:
        return self.reexport or self.colors is not None or Path(self.src).suffix.lower() in (".obj", ".fbx")


_CARS = "quaternius/realistic-car-pack/extrait/Realistic Car Pack - Nov 2018/OBJ"
_TRANSPORT = "quaternius/public-transport/extrait/quaternius_public_transport_glb_cc0_v1/source_reference"
_CITY_PACK = "polypizza/bundles/city-pack/extrait"
_ROADS = "kenney/city-kit-roads/extrait/Models/GLB format"

# Couleurs choisies par nous : les OBJ du Public Transport Pack n'en ont pas (tous les matériaux en gris 0,64).
# Noms des matériaux identifiés en les colorant un par un (carrosserie haute, basse, roues…).
_GLASS, _TIRE, _PLATE, _LAMP, _BUMPER = (0.12, 0.16, 0.2), (0.08, 0.08, 0.08), (0.9, 0.9, 0.85), (1.0, 0.9, 0.6), (0.2, 0.2, 0.22)
_BUS_COLORS = {"Top": (0.93, 0.93, 0.9), "Bottom": (0.75, 0.12, 0.12), "Bumper": _BUMPER, "Details": _PLATE,
               "Lights": _LAMP, "Material": _TIRE, "Windows": _GLASS}
_SCHOOL_BUS_COLORS = {"Yellow": (0.95, 0.72, 0.1), "Bumper": _BUMPER, "Details": _PLATE, "Lights": _LAMP,
                      "Wheel": _TIRE, "Windows": _GLASS}
_BICYCLE_COLORS = {"Bike": (0.1, 0.45, 0.6), "Handle": (0.1, 0.1, 0.1), "Material.003": (0.1, 0.1, 0.1), "Wheel": _TIRE}

SOURCES: tuple[AssetSource, ...] = (
    # Quaternius, Lowpoly Cars (zip « Realistic Car Pack - Nov 2018 ») : OBJ convertis par Blender
    AssetSource("quaternius/lowpoly-cars/normal-car-1.glb", f"{_CARS}/NormalCar1.obj"),
    AssetSource("quaternius/lowpoly-cars/normal-car-2.glb", f"{_CARS}/NormalCar2.obj"),
    AssetSource("quaternius/lowpoly-cars/suv.glb", f"{_CARS}/SUV.obj"),
    AssetSource("quaternius/lowpoly-cars/taxi.glb", f"{_CARS}/Taxi.obj"),
    AssetSource("quaternius/lowpoly-cars/sports-car.glb", f"{_CARS}/SportsCar.obj"),
    # Quaternius, Public Transport : les .glb du zip reconditionné (Eclair Assets) ont perdu leurs matériaux,
    # on convertit donc les OBJ + MTL d'origine qu'il contient (source_reference/).
    AssetSource("quaternius/public-transport/bus.glb", f"{_TRANSPORT}/Bus.obj", colors=_BUS_COLORS),
    AssetSource("quaternius/public-transport/school-bus.glb", f"{_TRANSPORT}/SchoolBus.obj", colors=_SCHOOL_BUS_COLORS),
    AssetSource("quaternius/public-transport/bicycle.glb", f"{_TRANSPORT}/Bicycle.obj", colors=_BICYCLE_COLORS),
    # Poly Pizza, modèles isolés (identifiant Poly Pizza dans le nom)
    AssetSource("polypizza/truck/truck-cXw6oiFtZ8.glb", "polypizza/truck/Truck by Quaternius - cXw6oiFtZ8.glb"),
    AssetSource("polypizza/truck/truckk-jHwRymyg2C.glb", "polypizza/truck/Truckk by KolosStudios - jHwRymyg2C.glb"),
    AssetSource("polypizza/bicycle/bicycle-19VoUuA2pcN.glb",
                "polypizza/bicycle/Bicycle by Poly by Google - 19VoUuA2pcN.glb"),
    AssetSource("polypizza/motorcycle/motorcycle-dse64pqMKAR.glb",
                "polypizza/motorcycle/Motorcycle by Poly by Google - dse64pqMKAR.glb"),
    # Poly Pizza, piétons animés de Quaternius (bundle « City Pack »). Exports FBX2glTF tels quels : panda3d-gltf
    # échoue (KeyError, peau déclarée avant son squelette) ou donne un squelette à la mauvaise échelle.
    # Blender les réécrit proprement, animations comprises.
    AssetSource("polypizza/person/man-HMnuH5geEG.glb", f"{_CITY_PACK}/Man.glb", reexport=True),
    AssetSource("polypizza/person/animated-woman-9kF7eTDbhO.glb", f"{_CITY_PACK}/Animated Woman.glb", reexport=True),
    AssetSource("polypizza/person/animated-woman-nIItLV9nxS.glb", f"{_CITY_PACK}/Animated Woman-nIItLV9nxS.glb",
                reexport=True),
    AssetSource("polypizza/person/animated-woman-qJ2gsTUBHL.glb", f"{_CITY_PACK}/Animated Woman-qJ2gsTUBHL.glb",
                reexport=True),
    # Kenney, City Kit (Roads) : les .glb partagent la texture d'atlas Textures/colormap.png
    AssetSource("kenney/city-kit-roads/traffic-light.glb", f"{_ROADS}/traffic-light.glb"),
    AssetSource("kenney/city-kit-roads/road-sign-stop.glb", f"{_ROADS}/road-sign-stop.glb"),
    AssetSource("kenney/city-kit-roads/road-sign-warning.glb", f"{_ROADS}/road-sign-warning.glb"),
    AssetSource("kenney/city-kit-roads/construction-cone.glb", f"{_ROADS}/construction-cone.glb"),
    AssetSource("kenney/city-kit-roads/light-square.glb", f"{_ROADS}/light-square.glb"),
    AssetSource("kenney/city-kit-roads/Textures/colormap.png", f"{_ROADS}/Textures/colormap.png"),
)


def find_blender() -> Path:
    candidates = [os.environ.get("BLENDER"), shutil.which("blender"), str(BLENDER_DEFAULT)]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Blender introuvable : installez-le ou définissez la variable d'environnement BLENDER")


def missing_sources(sources: tuple[AssetSource, ...] = SOURCES, downloads: Path = DOWNLOADS) -> list[AssetSource]:
    return [s for s in sources if not (downloads / s.src).is_file()]


def build(sources: tuple[AssetSource, ...] = SOURCES, downloads: Path = DOWNLOADS, out: Path = ASSETS_3D) -> list[Path]:
    """Copie ou convertit chaque source. Lève FileNotFoundError si un téléchargement manque."""
    missing = missing_sources(sources, downloads)
    if missing:
        raise FileNotFoundError("téléchargements absents : " + ", ".join(s.src for s in missing))
    blender = find_blender() if any(s.needs_blender for s in sources) else None
    written = []
    for source in sources:
        src, dst = downloads / source.src, out / source.dest
        dst.parent.mkdir(parents=True, exist_ok=True)
        if source.needs_blender:
            command = [str(blender), "-b", "--factory-startup", "-P", str(BLENDER_SCRIPT), "--", str(src), str(dst)]
            if source.colors:
                command.append(json.dumps(source.colors))
            subprocess.run(command, check=True, capture_output=True, text=True)
        else:
            shutil.copyfile(src, dst)
        written.append(dst)
        print(f"{'converti' if source.needs_blender else 'copié  '} {source.dest}")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="liste les téléchargements manquants")
    args = parser.parse_args(argv)
    if args.check:
        missing = missing_sources()
        for source in missing:
            print(f"manquant : _downloads/{source.src}")
        print(f"{len(SOURCES) - len(missing)} / {len(SOURCES)} sources présentes")
        return
    build()


if __name__ == "__main__":
    main()
