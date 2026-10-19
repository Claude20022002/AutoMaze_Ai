# AutoMaze AI — Module vidéo de reconnaissance visuelle

> **Document de cadrage destiné à Claude Code.** Lis ce document avant d'inspecter ou de modifier le dépôt. L'objectif est de développer un lecteur vidéo annoté pour le projet AutoMaze AI, sans compromettre la partie académique consacrée à l'entraînement et à l'évaluation d'un modèle de classification créé par l'équipe.

## 1. Objectif

Créer une application locale qui lit une vidéo filmée depuis une voiture et affiche les noms des objets visibles pendant la lecture. Exemple : `voiture`, `personne`, `vélo`, `moto`, `bus`, `camion`.

Le résultat attendu est une vidéo affichée dans une fenêtre ou une interface avec, si possible, un rectangle autour de chaque objet localisé et son étiquette. La lecture doit rester fluide autant que le matériel le permet. L'application doit aussi proposer un mode diagnostic indiquant la méthode de localisation utilisée, le nombre d'objets traités et le temps de traitement/FPS.

**Limite fondamentale :** un classifieur seul ne localise pas les objets dans une image complète. Le système doit donc séparer clairement :
1. la localisation approximative des régions pouvant contenir des objets ;
2. le recadrage de ces régions ;
3. la classification de chaque recadrage par notre modèle entraîné ;
4. l'affichage des étiquettes et scores sur la vidéo.

Le module vidéo ne doit pas prétendre que la localisation est fournie par le classifieur.

## 2. Cadre académique et règles non négociables

- **Ne pas utiliser YOLO**, aucun de ses dérivés ni aucun autre détecteur préentraîné.
- **Ne pas utiliser de modèle de classification préentraîné** (poids préentraînés, transfert d'apprentissage ou API distante de vision).
- Le modèle de classification final sera entraîné par l'équipe, à partir d'une initialisation aléatoire, dans les notebooks du projet.
- Ne pas inventer de poids, de résultats, de métriques ou de classes reconnues. Si le modèle entraîné n'est pas encore disponible, fournir un mode démo clairement marqué **SIMULATION — modèle réel non chargé** ; ne jamais présenter des étiquettes simulées comme des prédictions réelles.
- Toute méthode de localisation ajoutée doit être documentée et autorisée par les consignes du cours. Privilégier d'abord des méthodes classiques explicables : mouvement/soustraction d'arrière-plan, contours, segmentation couleur/texture ou fenêtres glissantes. Ces méthodes ne sont pas équivalentes à un détecteur moderne et peuvent être imprécises sur une route réelle.
- Si aucune méthode classique ne permet de localiser correctement les objets sur les vidéos tests, signaler honnêtement cette limite et demander confirmation avant d'introduire une autre architecture. Ne contourner aucune règle du professeur.
- Le module vidéo doit rester indépendant du code d'entraînement. Ne pas écraser les notebooks, le dataset ou les poids existants.
- Ne jamais lancer de suppression massive, de réinitialisation Git, de réécriture de l'historique ou de commande destructive. Inspecter le dépôt et faire un plan avant toute modification importante.

## 3. Classes cibles

Le modèle de classification du projet est prévu pour ces six classes :

| ID interne | Classe affichée | Nom anglais indicatif |
|---:|---|---|
| 0 | Personne | person |
| 1 | Vélo | bicycle |
| 2 | Voiture | car |
| 3 | Moto | motorcycle |
| 4 | Bus | bus |
| 5 | Camion | truck |

Les identifiants internes doivent être alignés sur le mapping réellement utilisé par le notebook d'entraînement et le fichier de métadonnées du modèle. **Ne pas supposer cet ordre sans vérifier le projet.** Une classe « autre/inconnu » ne doit pas être inventée comme si elle avait été entraînée ; elle peut être affichée par le logiciel quand la confiance est trop faible.

## 4. Architecture souhaitée

Concevoir des modules séparés, avec des interfaces simples :

```text
Fichier vidéo / webcam (facultatif)
              |
              v
       VideoSource
  (lecture, pause, reprise, fin)
              |
              v
       FrameProcessor
  (redimensionnement, cadence)
              |
              v
    ObjectLocalizer (abstrait)
  (méthode classique documentée)
              |
              v
  Regions / bounding boxes (x, y, w, h)
              |
              v
         CropPreprocessor
 (bornes, taille, canaux, normalisation)
              |
              v
     ClassifierInterface
 (modèle du projet, poids locaux)
              |
              v
       PredictionFilter
 (seuil, lissage temporel facultatif)
              |
              v
       VideoAnnotator
 (rectangle, classe, confiance)
              |
              v
      Fenêtre de lecture annotée
```

### Responsabilités

- `VideoSource` : ouvrir un chemin vidéo, vérifier les erreurs, lire les frames, conserver le ratio d'image et gérer la fin du fichier.
- `ObjectLocalizer` : retourner zéro ou plusieurs régions candidates avec coordonnées et score de localisation si disponible. Implémentation remplaçable, sans dépendance à un modèle préentraîné.
- `CropPreprocessor` : borner les coordonnées à l'image, rejeter les régions vides/trop petites, convertir BGR/RGB correctement et appliquer exactement le prétraitement attendu par le classifieur.
- `ClassifierInterface` : charger le modèle entraîné depuis un chemin local configurable ; valider les classes et la forme d'entrée ; renvoyer la classe prédite et les probabilités.
- `VideoAnnotator` : dessiner rectangles et étiquettes sans modifier l'image source utilisée pour l'inférence.
- `PlaybackController` : lecture, pause/reprise, quitter, avancer si simple à implémenter.
- `Metrics` : mesurer le temps de lecture et d'inférence séparément, FPS réel, nombre de régions candidates et nombre de prédictions acceptées.

Le choix du framework (probablement Python + OpenCV et PyTorch si c'est déjà le stack du dépôt) doit être confirmé après inspection du dépôt. Ne pas ajouter de grosses dépendances sans nécessité.

## 5. Localisation : approche progressive

La vidéo routière réelle est difficile : les objets sont de tailles différentes, partiellement cachés, mobiles, parfois flous, et la caméra bouge elle-même. Il faut donc avancer par étapes et mesurer les limites.

### Étape A — lecteur vidéo fonctionnel

- Ouvrir une vidéo locale (`.mp4`, `.avi`, formats compatibles OpenCV).
- Lire toutes les frames et les afficher.
- Montrer un message d'erreur compréhensible si le chemin est invalide ou le codec non pris en charge.
- Gérer la pause/reprise et la fermeture propre.

### Étape B — localisation classique expérimentale

Créer une interface `ObjectLocalizer` et au moins une méthode classique testable. Évaluer par exemple la soustraction d'arrière-plan ou les contours pour générer des régions candidates. Comme la caméra est mobile, expliquer dans la documentation que la soustraction d'arrière-plan standard peut échouer ; ne pas affirmer qu'elle détecte fiablement les voitures ou les personnes.

Une méthode de fenêtres glissantes peut être envisagée, mais ne pas lancer un balayage massif sur chaque frame sans contrôle du coût. Si elle est utilisée, documenter l'échelle, le pas et la cadence d'analyse.

### Étape C — classification des recadrages

- Charger uniquement le classifieur du projet entraîné localement.
- Appliquer le même prétraitement qu'à l'entraînement (taille, ordre des canaux, normalisation).
- Rejeter les recadrages invalides.
- Afficher `Inconnu` si la confiance est inférieure à un seuil configurable ; présenter ce seuil comme un paramètre à évaluer, pas comme une garantie.
- Ne pas confondre le score de classification avec une probabilité calibrée ou une certitude absolue.

### Étape D — stabilité vidéo et annotation

- Dessiner un rectangle et le nom de classe à proximité de chaque région acceptée.
- Ajouter la confiance en option.
- Réduire le scintillement des étiquettes par un lissage temporel simple, si cela n'associe pas incorrectement des objets différents.
- Si aucun objet n'est localisé, afficher la vidéo normalement sans fabriquer de prédictions.
- Éviter de traiter chaque frame si cela rend la lecture trop lente : permettre une cadence configurable (par exemple une frame sur 2 ou 3), tout en conservant une lecture compréhensible.

## 6. Mode d'intégration avec le modèle

Le module vidéo ne doit pas dépendre d'un nom de fichier supposé. Chercher d'abord dans le dépôt comment le modèle est sauvegardé et chargé. Si l'interface n'existe pas encore, proposer un contrat minimal, par exemple :

```python
class ClassifierInterface:
    def predict(self, crops):
        """Retourne une prédiction par crop : label, class_id, confidence."""
        raise NotImplementedError
```

Adapter ce contrat au framework réellement utilisé. Le modèle doit être chargé une seule fois au démarrage, et non à chaque frame. Si les poids ne sont pas présents, afficher une erreur explicite avec la commande ou l'étape manquante ; ne pas substituer un modèle externe.

Un fichier de métadonnées devrait documenter au minimum : noms et ordre des classes, taille d'entrée, normalisation, framework/version, date ou identifiant d'entraînement et chemin des poids.

## 7. Interface utilisateur minimale

Prévoir une interface simple, adaptée à une démonstration de projet :

- ouverture d'un fichier vidéo local ;
- lecture/pause ;
- fenêtre vidéo annotée ;
- indication claire de l'état du modèle : `Modèle chargé` ou `Modèle non disponible` ;
- affichage facultatif des FPS, du temps d'inférence, du nombre de régions et du mode de localisation ;
- message clair si aucune région n'est détectée ;
- sortie propre avec la touche `q` ou le bouton Fermer.

Ne pas enregistrer ou téléverser automatiquement les vidéos. Le traitement doit se faire localement par défaut. Ne pas ajouter de reconnaissance faciale ou d'identification de personnes : seule la catégorie générale `Personne` est dans le périmètre.

## 8. Organisation du dépôt proposée

Respecter la structure existante si elle est déjà différente. Ne pas déplacer arbitrairement les fichiers. Proposition si le dépôt n'a pas encore de structure dédiée :

```text
automaze-ai/
├── README.md
├── requirements.txt
├── configs/
│   └── video.yaml
├── src/
│   ├── video_app.py
│   ├── video_source.py
│   ├── localizers/
│   │   ├── base.py
│   │   └── classical.py
│   ├── classification/
│   │   ├── classifier_interface.py
│   │   └── model_loader.py
│   ├── preprocessing.py
│   ├── annotation.py
│   └── metrics.py
├── models/
│   └── README.md
├── tests/
│   ├── test_video_source.py
│   ├── test_preprocessing.py
│   └── test_classifier_interface.py
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_hog_svm.ipynb
│   └── 04_cnn_training.ipynb
└── data/  # données locales, normalement ignorées par Git
```

Ne pas committer de gros fichiers vidéo, le dataset COCO complet ou des poids lourds. Ajouter au `.gitignore` les chemins locaux appropriés, en vérifiant d'abord les règles existantes pour ne pas masquer des fichiers nécessaires.

## 9. Configuration et lancement

Prévoir une configuration centralisée plutôt que des constantes dispersées. Paramètres utiles :

- `video_path` : chemin du fichier, fourni à l'exécution de préférence ;
- `model_path` : chemin local des poids entraînés ;
- `metadata_path` : classes et prétraitement ;
- `input_size` : lu depuis les métadonnées du modèle si possible ;
- `confidence_threshold` : seuil configurable ;
- `process_every_n_frames` : cadence d'inférence ;
- `show_fps` : booléen ;
- `localizer` : méthode sélectionnée ;
- `min_box_width` / `min_box_height` : limites pour filtrer les régions minuscules.

Documenter les commandes exactes une fois le code réellement implémenté. À titre de forme souhaitée, la commande pourrait être :

```bash
python -m src.video_app --video "data/videos/test.mp4" --model "models/classifier.pt"
```

Ne pas présenter cette commande comme fonctionnelle avant de l'avoir testée et adapter le format du modèle au dépôt.

## 10. Tests et critères d'acceptation

Avant de déclarer le module terminé, vérifier :

1. Une vidéo valide se lit du début à la fin sans plantage.
2. Un chemin inexistant affiche une erreur utile.
3. La pause, la reprise et la fermeture fonctionnent.
4. Les coordonnées de régions sont toujours bornées à l'image.
5. Les recadrages vides ou trop petits sont ignorés proprement.
6. Le prétraitement est identique à celui de l'entraînement.
7. Le modèle est chargé une fois, et une absence de poids est signalée sans modèle de secours préentraîné.
8. Les classes affichées correspondent au mapping du modèle.
9. Une faible confiance produit `Inconnu` ou aucune étiquette selon la configuration, pas une classe inventée.
10. Les FPS et temps d'inférence mesurés sont calculés réellement.
11. La vidéo brute reste lisible quand aucune région n'est candidate.
12. Les tests et commandes sont documentés, et le module peut être utilisé sans exécuter les notebooks d'entraînement à chaque lancement.

Préparer une petite vidéo de test locale et des tests unitaires synthétiques. Ne pas télécharger ou intégrer des vidéos privées sans autorisation.

## 11. Protocole d'évaluation honnête

Tester sur plusieurs extraits : route avec circulation, piétons, vélos/motos si disponibles, scènes peu chargées, ombres et changements de lumière. Pour chaque extrait, noter :

- si l'objet a été localisé ou manqué ;
- si le rectangle contient principalement l'objet ;
- si la classe prédite est correcte ;
- les faux positifs (étiquette sur un élément qui n'est pas un objet cible) ;
- FPS moyen et latence approximative ;
- conditions où la méthode échoue.

Ne pas évaluer seulement la classification sur des crops propres et conclure que la détection en vidéo fonctionne. Les performances de localisation et celles de classification doivent être discutées séparément.

## 12. Plan de travail attendu de Claude Code

Procède dans cet ordre :

1. Inspecter le dépôt : arborescence, environnement, dépendances, notebooks, mapping de classes, format du modèle et état Git.
2. Résumer l'existant et identifier ce qui est réutilisable. Ne modifier aucun fichier avant ce diagnostic et un plan court.
3. Implémenter le lecteur vidéo minimal, sans modèle ni localisation, puis le tester.
4. Ajouter une interface de localisation et une méthode classique clairement documentée, sans modèle préentraîné.
5. Ajouter l'adaptateur vers le classifieur local du projet. Si le modèle n'est pas encore disponible, garder une interface explicite et testable.
6. Ajouter l'annotation, les métriques, les paramètres et les tests.
7. Mettre à jour ce README avec les vraies commandes, dépendances, limites observées et résultats réellement mesurés.
8. À la fin, fournir un résumé des fichiers modifiés, commandes de lancement testées, résultats des tests, problèmes connus et étapes manuelles restantes.

Si un choix affecte les règles académiques, la portée du projet, ou implique une dépendance préentraînée, **s'arrêter et demander confirmation** plutôt que de décider seul.

## 13. Limites à communiquer pendant la présentation

Une caméra embarquée filme une scène entière et bouge continuellement. Des méthodes classiques de localisation peuvent manquer des objets, confondre des éléments de la route avec des objets ou produire des rectangles instables. Le classifieur entraîné sur des objets recadrés peut aussi échouer sur des objets lointains, flous, partiellement cachés ou très petits.

Le projet est une démonstration pédagogique de classification intégrée à un flux vidéo, **pas un système de conduite autonome sûr**. Il ne doit pas être utilisé pour prendre des décisions réelles de conduite.
