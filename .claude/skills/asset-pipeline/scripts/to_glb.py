"""Convertit un modèle 3D (OBJ, FBX, glTF) en .glb avec Blender, sans interface.

    blender -b --factory-startup -P .claude/skills/asset-pipeline/scripts/to_glb.py -- entree.fbx sortie.glb

Testé le 31 octobre 2026 avec Blender 5.2 : un OBJ ou un FBX Y-up devient un .glb que Panda3D charge
à l'endroit (Z vers le haut). Le .glb garde les matériaux ; les textures sont intégrées au fichier.
"""

import sys
from pathlib import Path

import bpy

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(args) != 2:
    raise SystemExit("usage : blender -b --factory-startup -P to_glb.py -- entree sortie.glb")
src, dst = Path(args[0]), Path(args[1])

bpy.ops.wm.read_factory_settings(use_empty=True)   # scène vide : pas de cube ni de caméra par défaut
ext = src.suffix.lower()
if ext == ".obj":
    bpy.ops.wm.obj_import(filepath=str(src))
elif ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=str(src))
elif ext in (".gltf", ".glb"):
    bpy.ops.import_scene.gltf(filepath=str(src))
else:
    raise SystemExit(f"format non géré : {ext}")

dst.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(filepath=str(dst), export_format="GLB")
print("OK", dst, "objets :", [o.name for o in bpy.data.objects])
