# Modèles 3D du simulateur : sources et licences

Ces modèles servent **uniquement à l'affichage** de la vue 3D. Ils ne sont jamais donnés au classifieur, ni utilisés
comme données d'entraînement ou de test du modèle des étudiants (voir `.claude/skills/sim3d-architecture`).

Chaque fichier est produit par `python -m simulator.render3d.build_assets` à partir des téléchargements listés ici.
Le choix des modèles par classe, leur taille et leur orientation sont dans `catalog.json`.

## Crédits à afficher (licences CC-BY : obligatoire)

- « Bicycle » par Poly by Google, https://poly.pizza/m/19VoUuA2pcN, licence CC-BY 3.0
- « Motorcycle » par Poly by Google, https://poly.pizza/m/dse64pqMKAR, licence CC-BY 3.0
- « Truckk » par KolosStudios, https://poly.pizza/m/jHwRymyg2C, licence CC-BY 3.0

Licence CC-BY 3.0 : https://creativecommons.org/licenses/by/3.0/

## Modèles CC0 (crédit non obligatoire, donné quand même)

| Auteur | Source | Fichiers | Licence vérifiée |
| --- | --- | --- | --- |
| Quaternius | Lowpoly Cars, https://quaternius.itch.io/lowpoly-cars (zip « Realistic Car Pack - Nov 2018 », OBJ convertis en .glb par Blender 5.2) | `quaternius/lowpoly-cars/*.glb` | `License.txt` du zip : CC0 1.0 |
| Quaternius | Public Transport Pack, https://quaternius.com/packs/publictransport.html | `quaternius/public-transport/*.glb` | page du pack : CC0. OBJ issus de la version reconditionnée par « Eclair Assets » (non officielle), sans couleurs : couleurs ajoutées par nous (`build_assets.py`) |
| Quaternius | Truck, https://poly.pizza/m/cXw6oiFtZ8 | `polypizza/truck/truck-cXw6oiFtZ8.glb` | page Poly Pizza : CC0 |
| Quaternius | Man, https://poly.pizza/m/HMnuH5geEG | `polypizza/person/man-HMnuH5geEG.glb` | page Poly Pizza : CC0 (identifié par le nom et les animations) |
| Quaternius | Animated Woman, https://poly.pizza/m/9kF7eTDbhO | `polypizza/person/animated-woman-9kF7eTDbhO.glb` | page Poly Pizza : CC0 (identifié par élimination) |
| Quaternius | Animated Woman, https://poly.pizza/m/nIItLV9nxS | `polypizza/person/animated-woman-nIItLV9nxS.glb` | page Poly Pizza : CC0 |
| Quaternius | Animated Woman, https://poly.pizza/m/qJ2gsTUBHL | `polypizza/person/animated-woman-qJ2gsTUBHL.glb` | page Poly Pizza : CC0 |
| Kenney | City Kit (Roads) 2.2, https://www.kenney.nl/assets/city-kit-roads | `kenney/city-kit-roads/*` | `License.txt` du zip : CC0 |

Licence CC0 1.0 : https://creativecommons.org/publicdomain/zero/1.0/

Sources vérifiées le 31 octobre 2026.
