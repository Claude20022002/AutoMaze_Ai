---
name: asset-pipeline
description: Récupérer, trier, convertir, normaliser et créditer les modèles 3D (véhicules, piétons, panneaux, feux) du simulateur 3D d'AutoMaze AI, venant de Kenney, Quaternius, Poly Pizza ou Sketchfab. À utiliser avant de télécharger ou d'ajouter un modèle 3D, de modifier simulator/assets/3d/, catalog.json, CREDITS.md ou simulator/render3d/build_assets.py, ou de changer les crédits affichés.
---

# Pipeline des modèles 3D

## Où sont les choses

| Élément | Rôle |
| --- | --- |
| `simulator/assets/3d/_downloads/` | Zips téléchargés et leur `extrait/`, **ignoré par Git**. `INVENTAIRE.md` y décrit chaque pack et son statut |
| `simulator/assets/3d/_local/` | Fichiers non redistribuables (Mixamo), **ignoré par Git** |
| `simulator/render3d/build_assets.py` | `SOURCES` : d'où vient chaque fichier commité (copie, conversion OBJ, réexport, couleurs) |
| `simulator/render3d/blender_to_glb.py` | Script exécuté par Blender : OBJ/FBX/glTF -> .glb, couleurs facultatives par matériau |
| `simulator/assets/3d/catalog.json` | « Blueprint library » : classe ou rôle -> modèles, taille réelle, cap, licence, animations |
| `simulator/assets/3d/CREDITS.md` | Auteurs, URL, licences ; crédits CC-BY à afficher |
| `simulator/render3d/catalog.py` | Lecture validée du catalogue ; `load_model` normalise (échelle, cap, sol) |
| `simulator/render3d/contact_sheet.py` | Planches de contrôle dans `reports/simulator/` |
| `tests/simulator/render3d/test_catalog.py` | Classes, fichiers, traçabilité, crédits, tailles réelles, animations |

## Ajouter ou remplacer un modèle

1. **Licence d'abord.** La lire sur la page du modèle ou dans le zip. Ordre de préférence : **CC0**, puis **CC-BY**
   (crédit obligatoire), puis CC-BY-SA seulement avec l'accord de l'utilisateur. Refuser NC, ND et les modèles payants
   (Sketchfab Store), sauf accord explicite et noté. Un bundle Poly Pizza **ne porte aucune licence** : vérifier
   chaque modèle sur sa page (`https://poly.pizza/m/<id>`) ; les auteurs y sont mélangés. Ne jamais inventer un nom
   de pack, d'auteur ni de licence.
2. Mettre le zip dans `_downloads/<source>/<pack>/` et l'extraire dans `extrait/`, en refusant tout chemin qui sort
   du dossier (zip slip). Ne lancer aucun script contenu dans un téléchargement.
3. Ajouter une ligne à `SOURCES` dans `build_assets.py`, puis lancer `python -m simulator.render3d.build_assets`
   (`--check` liste ce qui manque). Blender : variable `BLENDER`, sinon PATH, sinon
   `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`.
4. Ajouter l'entrée dans `catalog.json` : `length_m` (véhicules) **ou** `height_m` (personnes, panneaux), `title`,
   `author`, `license`, `url`, puis `animations` (`walk`, `idle`) pour un personnage. Compléter `CREDITS.md`.
5. Trouver le cap avec les planches (ci-dessous), le mettre dans `heading`, régénérer.
6. `python -m pytest tests/simulator/render3d`.

Ne jamais modifier un fichier source à la main : échelle, cap et origine se corrigent dans `catalog.json`.
On ne commite que les .glb produits, jamais les zips, les OBJ/FBX sources ni les caches .bam.

## Planches de contrôle (à regarder après chaque changement)

```powershell
python -m simulator.render3d.contact_sheet --view fronts   # planche_assets_avants.png
python -m simulator.render3d.contact_sheet --view top      # planche_assets_dessus.png
python -m simulator.render3d.contact_sheet                 # planche_assets.png (perspective)
```

- **fronts** : chaque modèle seul, vu de trois quarts depuis +Y. Un modèle bien orienté montre **sa face avant**
  (phares, visage). Si on voit l'arrière : `heading: 180`. Si l'avant est à gauche de l'image (vers -X) :
  `heading: -90`. S'il est à droite (vers +X) : `heading: 90`.
- **top** : la flèche rouge (+Y) doit sortir de l'avant de chaque véhicule ; tous les véhicules sont posés en long selon Y.
- Signalisation (Kenney) : la face des panneaux regarde **-Y** (on voit leur dos depuis +Y). Observé, à
  confirmer au placement sur la carte (skill `osm-to-3d`).

## Pièges rencontrés (31 octobre 2026)

| Symptôme | Cause | Correction |
| --- | --- | --- |
| OBJ couché dans Panda3D | `p3assimp` ne convertit pas Y-up en Z-up | Convertir en .glb avec Blender (OBJ dans `SOURCES`) |
| `KeyError` dans `gltf/_converter.py` au chargement d'un personnage | Export FBX2glTF : peau déclarée avant son squelette | `reexport=True` : Blender réécrit le .glb, animations comprises |
| Personnage géant, minuscule ou couché sur la planche | Bornes au repos fausses (T-pose, squelette à l'échelle 100) | `load_model` mesure les bornes **dans la pose `idle`** ; réexport Blender |
| Modèle gris uniforme | Matériaux sans couleur (tous les `Kd 0.64`, par exemple Public Transport reconditionné) | `colors={matériau: (r, g, b)}` dans `SOURCES` ; identifier les matériaux en les colorant un par un |
| Modèle blanc (Kenney) | Texture d'atlas `Textures/colormap.png` absente | La copier avec le même chemin relatif (déjà dans `SOURCES`) |

## Animations des personnes

- Les piétons actuels (Quaternius via Poly Pizza, CC0) ont **leurs propres animations** (`walk`, `idle` dans
  `catalog.json`). Ils n'ont besoin ni de Mixamo ni de retargeting.
- Quaternius Universal Base Characters (version gratuite : 2 personnages en tenue de super-héros) et Universal
  Animation Library (43 animations CC0, squelette `root/pelvis/spine_01…`) sont téléchargés mais pas utilisés.
- **Mixamo** : interdit de redistribuer les fichiers bruts et de s'en servir pour entraîner un modèle (FAQ Adobe
  de 2022, à revérifier). Tout fichier Mixamo ou dérivé va dans `_local/`, jamais dans Git. L'application doit
  fonctionner sans : personnage immobile.

## Crédits

- Kenney et Quaternius en CC0 : le crédit n'est pas obligatoire, on crédite quand même. Les CC-BY de `CREDITS.md`
  doivent apparaître à l'écran de l'application et dans le rapport, comme « © contributeurs OpenStreetMap » (ODbL).
- Les modèles 3D ne sont **jamais** des données d'entraînement ni de test du modèle des étudiants, et le
  classifieur ne les reçoit pas (skill `sim3d-architecture`).
