"""Convertit un modèle 3D (OBJ, FBX, glTF) en .glb. Script exécuté PAR Blender, pas par Python directement :

    blender -b --factory-startup -P simulator/render3d/blender_to_glb.py -- entree.obj sortie.glb ['{"Top": [1, 1, 1]}']

Testé avec Blender 5.2 : un OBJ ou un FBX Y-up devient un .glb que Panda3D charge à l'endroit (Z vers le haut).
Le .glb garde les matériaux ; les textures éventuelles sont intégrées au fichier.
Troisième argument facultatif : couleurs (RGB entre 0 et 1) à donner aux matériaux nommés, pour les modèles livrés
sans couleurs.
"""

import json
import sys
from pathlib import Path

import bpy

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(args) not in (2, 3):
    raise SystemExit("usage : blender -b --factory-startup -P blender_to_glb.py -- entree sortie.glb [couleurs_json]")
src, dst = Path(args[0]), Path(args[1])
colors = json.loads(args[2]) if len(args) == 3 else {}

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

for name, rgb in colors.items():
    material = bpy.data.materials.get(name)
    if material is None:
        raise SystemExit(f"matériau absent : {name} (présents : {[m.name for m in bpy.data.materials]})")
    principled = material.node_tree.nodes.get("Principled BSDF") if material.use_nodes else None
    if principled is not None:
        principled.inputs["Base Color"].default_value = (*rgb, 1.0)
    material.diffuse_color = (*rgb, 1.0)

dst.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(filepath=str(dst), export_format="GLB")
print("OK", dst, "objets :", [o.name for o in bpy.data.objects])
