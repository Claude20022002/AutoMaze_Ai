"""Tests de la banque d'obstacles et des images de remplacement."""

import numpy as np
import pytest
from PIL import Image

from simulator.contracts import CLASSES, Obstacle
from simulator.perception.images import PLACEHOLDER_COLORS, check_image_rgb, load_rgb, placeholder_image
from simulator.perception.obstacle_bank import ObstacleBank

EDGE = (1, 2, 0)


@pytest.fixture
def crops_dir(tmp_path):
    """Faux jeu de test : 3 images pour « car », 1 pour « person », rien pour les autres classes."""
    for label, n in (("car", 3), ("person", 1)):
        folder = tmp_path / label
        folder.mkdir()
        for i in range(n):
            Image.new("RGB", (20 + i, 30), (10 * i, 100, 200)).save(folder / f"{i}.jpg")
    (tmp_path / "car" / "notes.txt").write_text("pas une image", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("label", CLASSES)
def test_placeholder_respects_contract(label):
    image = placeholder_image(label)
    check_image_rgb(image)
    assert image.shape == (128, 128, 3)
    assert tuple(image[0, 0]) == PLACEHOLDER_COLORS[label]  # coin de l'image = couleur de fond
    assert (image == 255).all(axis=2).any()                 # du texte blanc est bien écrit


def test_placeholder_colors_are_distinct():
    assert len(set(PLACEHOLDER_COLORS.values())) == len(CLASSES)
    assert set(PLACEHOLDER_COLORS) == set(CLASSES)


def test_placeholder_rejects_unknown_label():
    with pytest.raises(ValueError):
        placeholder_image("unknown")


def test_empty_bank_uses_placeholders(tmp_path):
    bank = ObstacleBank(tmp_path / "absent")
    assert not bank.has_real_images
    rng = np.random.default_rng(0)
    for label in CLASSES:
        assert bank.count(label) == 0
        assert bank.sample_image_path(label, rng) is None
    image = bank.load_image(Obstacle(edge=EDGE, true_label="bus", image_path=None))
    assert np.array_equal(image, placeholder_image("bus"))


def test_bank_indexes_images_only(crops_dir):
    bank = ObstacleBank(crops_dir)
    assert bank.count("car") == 3          # le .txt est ignoré
    assert bank.count("person") == 1
    assert bank.count("bus") == 0
    assert not bank.has_real_images        # certaines classes n'ont pas d'image


def test_sample_returns_image_of_requested_class(crops_dir):
    bank = ObstacleBank(crops_dir)
    rng = np.random.default_rng(0)
    paths = {bank.sample_image_path("car", rng) for _ in range(50)}
    assert paths == {crops_dir / "car" / f"{i}.jpg" for i in range(3)}
    assert bank.sample_image_path("bus", rng) is None  # classe sans image : remplacement


def test_sampling_is_reproducible(crops_dir):
    bank = ObstacleBank(crops_dir)
    draw = lambda seed: [bank.sample_image_path("car", np.random.default_rng(seed)) for _ in range(1)]  # noqa: E731
    seq = lambda seed: [bank.sample_image_path("car", rng) for rng in [np.random.default_rng(seed)] for _ in range(20)]  # noqa: E731
    assert draw(5) == draw(5)
    assert seq(5) == seq(5)


def test_load_real_image_is_rgb_uint8(crops_dir):
    bank = ObstacleBank(crops_dir)
    path = crops_dir / "person" / "0.jpg"
    image = bank.load_image(Obstacle(edge=EDGE, true_label="person", image_path=path))
    check_image_rgb(image)
    assert image.shape == (30, 20, 3)  # (H, W, 3)


def test_load_rgb_converts_grayscale_and_rgba(tmp_path):
    Image.new("L", (8, 6), 128).save(tmp_path / "gris.png")
    Image.new("RGBA", (8, 6), (255, 0, 0, 128)).save(tmp_path / "rgba.png")
    for name in ("gris.png", "rgba.png"):
        image = load_rgb(tmp_path / name)
        check_image_rgb(image)
        assert image.shape == (6, 8, 3)


def test_load_rgb_keeps_channel_order(tmp_path):
    """Piège BGR/RGB : un pixel rouge doit rester rouge (canal 0)."""
    Image.new("RGB", (4, 4), (255, 0, 0)).save(tmp_path / "rouge.png")
    assert tuple(load_rgb(tmp_path / "rouge.png")[0, 0]) == (255, 0, 0)


def test_unknown_label_rejected(crops_dir):
    with pytest.raises(ValueError):
        ObstacleBank(crops_dir).sample_image_path("tram", np.random.default_rng(0))
