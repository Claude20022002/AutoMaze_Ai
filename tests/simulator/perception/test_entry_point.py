"""Tests du point d'entrée utilisé par l'équipe A : make_classifier et make_image_provider (plan, section 3)."""

import numpy as np
import pytest
from PIL import Image

from simulator.config import SimConfig
from simulator.contracts import CLASSES, Obstacle
from simulator.perception import CLASSIFIER_MODES, make_classifier, make_image_provider
from simulator.perception.images import PLACEHOLDER_NOISE, check_image_rgb, placeholder_image
from simulator.perception.simulated import SimulatedClassifier


def _obstacle(label: str = "car", edge=(10, 20, 0), offset_m: float = 12.5) -> Obstacle:
    return Obstacle(edge=edge, true_label=label, image_path=None, offset_m=offset_m)


@pytest.fixture
def crops_dir(tmp_path):
    """Faux jeu de test : 10 images de voitures, toutes différentes, aucune pour les autres classes."""
    folder = tmp_path / "car"
    folder.mkdir()
    for i in range(10):
        Image.new("RGB", (16, 16), (i * 20, 0, 0)).save(folder / f"{i}.png")
    return tmp_path


def test_make_classifier_uses_config():
    clf = make_classifier("simulation", SimConfig(seed=11, simulated_error_rate=0.0, confidence_threshold=0.3))
    assert isinstance(clf, SimulatedClassifier)
    assert (clf.seed, clf.error_rate, clf.threshold) == (11, 0.0, 0.3)


def test_make_classifier_modes():
    assert CLASSIFIER_MODES == ("simulation", "reference", "ours")
    for name in ("reference", "ours"):
        with pytest.raises(NotImplementedError):  # aucun modèle de secours
            make_classifier(name, SimConfig())
    with pytest.raises(ValueError):
        make_classifier("yolo", SimConfig())


def test_provider_without_crops_gives_placeholders(tmp_path):
    images = make_image_provider(SimConfig(), crops_dir=tmp_path / "absent")
    for label in CLASSES:
        image = images(_obstacle(label))
        check_image_rgb(image)
        difference = np.abs(image.astype(int) - placeholder_image(label).astype(int))
        assert difference.max() <= PLACEHOLDER_NOISE  # l'image de remplacement, au léger bruit près


def test_placeholders_differ_between_obstacles_of_same_class(tmp_path):
    """Sinon le mode simulation donnerait la même réponse pour tous les obstacles d'une même classe."""
    images = make_image_provider(SimConfig(), crops_dir=tmp_path / "absent")
    seen = {images(_obstacle("car", edge=(i, i + 1, 0))).tobytes() for i in range(20)}
    assert len(seen) == 20


def test_simulated_errors_vary_across_placeholder_obstacles(tmp_path):
    config = SimConfig(simulated_error_rate=0.5)
    images, clf = make_image_provider(config, crops_dir=tmp_path / "absent"), make_classifier("simulation", config)
    labels = {clf.predict(images(_obstacle("car", edge=(i, 0, 0))), true_label="car").label for i in range(40)}
    assert "car" in labels and len(labels) > 1


def test_same_obstacle_same_image(crops_dir):
    first = make_image_provider(SimConfig(seed=3), crops_dir=crops_dir)
    second = make_image_provider(SimConfig(seed=3), crops_dir=crops_dir)
    obstacle = _obstacle()
    assert np.array_equal(first(obstacle), first(obstacle))
    assert np.array_equal(first(obstacle), second(obstacle))  # indépendant de l'ordre des appels


def test_different_obstacles_use_several_images(crops_dir):
    images = make_image_provider(SimConfig(seed=3), crops_dir=crops_dir)
    seen = {images(_obstacle(edge=(i, i + 1, 0))).tobytes() for i in range(30)}
    assert len(seen) > 3


def test_provider_uses_crops_and_falls_back_per_class(crops_dir):
    images = make_image_provider(SimConfig(), crops_dir=crops_dir)
    assert images(_obstacle("car")).shape == (16, 16, 3)                            # vrai recadrage
    assert images(_obstacle("bus")).shape == placeholder_image("bus").shape        # pas de recadrage : remplacement


def test_provider_respects_given_image_path(crops_dir):
    images = make_image_provider(SimConfig(), crops_dir=crops_dir)
    path = crops_dir / "car" / "7.png"
    image = images(Obstacle(edge=(1, 2, 0), true_label="car", image_path=path))
    assert tuple(image[0, 0]) == (140, 0, 0)
