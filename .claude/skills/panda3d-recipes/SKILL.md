---
name: panda3d-recipes
description: Recettes Panda3D vérifiées sur le PC du projet (Windows 11, GPU AMD intégré, Python 3.13) pour le simulateur 3D d'AutoMaze AI : installation, fenêtre ou mode sans fenêtre, capteur caméra hors écran vers numpy RGB, projection des boîtes 3D pour recadrer un obstacle, chargement glTF/OBJ, simplepbr, boucle à pas fixe, caméras, textes accentués, tests pytest. À utiliser pour écrire ou déboguer du code qui importe panda3d.
---

# Panda3D : recettes vérifiées pour AutoMaze AI

Toutes les recettes marquées **(testé)** ont tourné le 31 octobre 2026 sur le PC du projet :
Panda3D 1.10.16, `wglGraphicsPipe`, OpenGL 4.6 sur « AMD Radeon(TM) Graphics ». Le reste est une indication
à vérifier au premier usage. L'architecture d'ensemble est dans le skill `sim3d-architecture`.

## Installation

```text
panda3d>=1.10.16          # wheel cp313-win_amd64 disponible
panda3d-gltf>=1.3         # chargeur .gltf/.glb (s'enregistre tout seul auprès du loader)
panda3d-simplepbr>=0.13   # rendu PBR des matériaux glTF ; facultatif sur GPU intégré
```

Ces lignes vont dans `requirements.txt`, fichier partagé : noter le changement dans le journal du plan.

## Démarrage : fenêtre, ou mode sans fenêtre pour les tests (testé)

```python
from panda3d.core import loadPrcFileData

# Toujours AVANT d'importer ShowBase : la configuration est lue à sa création.
loadPrcFileData("", "window-type offscreen\naudio-library-name null\nsync-video false")
from direct.showbase.ShowBase import ShowBase

base = ShowBase()
```

- Les valeurs de `window-type` :
  - `onscreen` : l'application normale ;
  - `offscreen` : rendu possible sans fenêtre visible, à utiliser pour les tests et les captures ;
  - `none` : aucun rendu, chargement de modèles seulement.
- **Une seule `ShowBase` par processus.** Dans pytest, utiliser une fixture `scope="session"`, puis
  `base.destroy()` à la fin.
- Avertissement sans conséquence : `FrameBufferProperties available less than requested` (le pilote donne plus que demandé).
- Ne jamais appeler `base.run()` dans un test. Pour avancer d'une image : `base.graphicsEngine.renderFrame()`, ou
  `base.taskMgr.step()` si les tâches doivent tourner.

## Capteur caméra RGB vers numpy (testé)

Équivalent du `sensor.camera.rgb` de CARLA. Résultat : `np.ndarray` uint8 de forme (H, W, 3) en **RGB**,
la ligne 0 en haut : exactement le contrat de `Classifier.predict`.

```python
import numpy as np
from panda3d.core import (FrameBufferProperties, GraphicsOutput, GraphicsPipe, Texture, WindowProperties)

def make_rgb_sensor(base, width: int, height: int, fov_deg: float = 60.0):
    fb = FrameBufferProperties()
    fb.setRgbColor(True)
    fb.setRgbaBits(8, 8, 8, 8)
    fb.setDepthBits(24)
    buf = base.graphicsEngine.makeOutput(
        base.pipe, "capteur_rgb", -2, fb, WindowProperties.size(width, height),
        GraphicsPipe.BF_refuse_window, base.win.getGsg(), base.win,
    )
    tex = Texture()
    buf.addRenderTexture(tex, GraphicsOutput.RTMCopyRam)   # copie en RAM à chaque image (coût mesurable)
    cam = base.makeCamera(buf)
    cam.node().getLens().setFov(fov_deg)
    return buf, tex, cam

def read_rgb(tex: Texture) -> np.ndarray:
    data = tex.getRamImageAs("RGB")                        # Panda3D stocke en BGRA : getRamImageAs remet en RGB
    img = np.frombuffer(data, np.uint8).reshape(tex.getYSize(), tex.getXSize(), 3)
    return np.ascontiguousarray(img[::-1])                 # OpenGL : origine en bas, on retourne verticalement
```

Pièges :
- Lire la texture **après** `renderFrame()`. La première image peut être vide, donc rendre deux fois au démarrage.
- Ne jamais lire `getRamImage()` brut sans le convertir : l'ordre BGRA inverse les canaux sans aucun message d'erreur.
- Désactiver le capteur quand il ne sert pas (`buf.setActive(False)`) : la copie en RAM coûte à chaque image.

## Vérité terrain : boîte 3D projetée, puis recadrage (testé, au pixel près)

Équivalent des boîtes englobantes de CARLA. On n'a besoin d'aucun détecteur, puisque le simulateur connaît l'obstacle.

```python
from panda3d.core import Point2, Point3

def project_bbox(cam, lens, node, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Boîte (x0, y0, x1, y1) en pixels de l'image retournée par read_rgb ; None si l'objet est hors champ."""
    lo, hi = node.getTightBounds(node)
    xs, ys = [], []
    for corner in (Point3(x, y, z) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)):
        p2 = Point2()
        if lens.project(cam.getRelativePoint(node, corner), p2):   # False si le coin sort du champ
            xs.append((p2.x + 1) / 2 * width)
            ys.append((1 - p2.y) / 2 * height)
    if not xs:
        return None
    x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
    x1, y1 = min(width, int(max(xs)) + 1), min(height, int(max(ys)) + 1)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
```

Recadrage : `img[y0:y1, x0:x1]`. Ajouter une marge d'environ 10 %, comme les recadrages du jeu de données,
et le rendre carré si le notebook 04 le fait. Un objet partiellement caché par un autre reste dans la boîte :
c'est voulu, la vraie caméra aurait le même problème.

## Modèles : glTF, OBJ, FBX (testé)

```python
from panda3d.core import Filename

model = base.loader.loadModel(Filename.fromOsSpecific(str(path)))  # chemin Windows avec espaces et accents
```

- **glTF/GLB** (via `panda3d-gltf`) : l'axe Y-up du glTF est **converti automatiquement** en Z-up. C'est le
  format à privilégier.
- **OBJ/FBX** (via le greffon `p3assimp`, inclus dans la wheel) : se chargent, mais **sans conversion d'axe**. Un
  modèle Y-up apparaît couché et doit être pivoté (`setP(90)`). Mieux vaut convertir en .glb (skill `asset-pipeline`).
- Repère Panda3D : X à droite, Y vers l'avant, Z vers le haut, en mètres. Le cap `H` est en degrés, 0 = +Y,
  dans le sens inverse des aiguilles d'une montre vu de dessus.
- Cache : `model.writeBamFile(...)`, puis rechargement en .bam, beaucoup plus rapide. Les .bam vont dans un dossier ignoré par Git.
- Animations des personnages glTF : `from direct.actor.Actor import Actor` avec `Actor(path)`, puis
  `actor.getAnimNames()` et `actor.loop("Walk")`. Les noms dépendent du pack : toujours les lister d'abord.

## Rendu PBR (testé)

```python
import simplepbr
pipeline = simplepbr.init()     # après ShowBase, une seule fois ; fonctionne sur le GPU intégré
```

Prévoir `--no-pbr`, qui n'appelle pas `simplepbr.init()` et ajoute une lumière ambiante et une lumière
directionnelle, au cas où le nombre d'images par seconde baisse. Avertissement normal pour un maillage sans
matériau : `primitive with no material, using an empty RenderState`.

## Boucle à pas fixe (mode synchrone de CARLA)

```python
from direct.task import Task

class SimLoop:
    def __init__(self, base, episode, step_seconds: float, speedup: float = 1.0) -> None:
        self.episode, self.dt, self.speedup, self.acc = episode, step_seconds, speedup, 0.0
        base.taskMgr.add(self.tick, "sim_tick")

    def tick(self, task):
        self.acc += globalClock.getDt() * self.speedup   # globalClock : variable globale de Panda3D
        while self.acc >= self.dt and not self.episode.done:
            self.episode.step()                          # la logique avance par pas entiers, comme sans affichage
            self.acc -= self.dt
        alpha = self.acc / self.dt                       # fraction du pas : interpolation de l'affichage
        # ... placer la voiture entre l'ancienne et la nouvelle position avec alpha
        return Task.cont
```

## Caméras et environnement

- Caméra de poursuite : `base.disableMouse()`, puis à chaque image `base.camera.setPos(car, 0, -12, 5)` et
  `base.camera.lookAt(car, 0, 8, 1)`. Vue conducteur : `setPos(car, 0.35, 0.2, 1.2)`.
- Ciel : `base.setBackgroundColor(...)`. Brouillard : `Fog("brume")`, `setExpDensity(...)`, puis
  `render.setFog(fog)`. Limiter `base.camLens.setFar(...)` à la portée du brouillard.
- Lumières : `AmbientLight` et `DirectionalLight`, attachées avec `render.setLight(np)`.

## Textes et interface 2D

- `OnscreenText` et `DirectGui` dessinent sur `aspect2d`, une couche 2D par-dessus la scène.
- Accents (« Vélo », « Arrêt prudent ») : charger une police qui les contient,
  `base.loader.loadFont(<DejaVuSans.ttf>)`. Le chemin se trouve avec
  `matplotlib.font_manager.findfont("DejaVu Sans")`, comme dans `perception/images.py`.
- Image numpy affichée en vignette : `Texture.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb)`,
  puis `tex.setRamImageAs(img[::-1].tobytes(), "RGB")` (retournement inverse de `read_rgb`) et une `OnscreenImage`.

## Performances (GPU intégré)

- `base.setFrameRateMeter(True)` pour mesurer d'abord. `render.analyze()` donne le nombre de Geoms et d'états.
- Beaucoup d'objets statiques : les regrouper sous un nœud, puis appeler `flattenStrong()` une fois après construction.
- Nombreux exemplaires d'un même modèle (arbres, lampadaires) : `instanceTo` partage la géométrie.
- Pas d'ombres portées par défaut. Textures en petite taille.

## Tests pytest

```python
import pytest
panda3d = pytest.importorskip("panda3d")      # le reste de la suite tourne même sans Panda3D

@pytest.fixture(scope="session")
def base():
    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "window-type offscreen\naudio-library-name null")
    from direct.showbase.ShowBase import ShowBase
    b = ShowBase()
    yield b
    b.destroy()
```

- Tester la géométrie pure (maillages de routes, projections, interpolation) **sans** ShowBase : c'est plus rapide et plus fiable.
- Test de concordance capteur : une boîte de couleur connue devant le capteur, puis vérifier la couleur au centre et la
  boîte projetée contre les pixels (recette ci-dessus).
