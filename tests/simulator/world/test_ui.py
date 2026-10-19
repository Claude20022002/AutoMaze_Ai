"""Test de l'interface pygame sans fenêtre (pilote vidéo « dummy »)."""

import pytest

from simulator.config import MAP_FILE

pytestmark = pytest.mark.skipif(not MAP_FILE.exists(), reason="carte absente")


def test_screenshot_renders(tmp_path, monkeypatch):
    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from simulator.main import main

    out = tmp_path / "capture.png"
    main(["--screenshot", str(out), "--steps", "120", "--episode", "1"])
    assert out.exists() and out.stat().st_size > 10_000


def test_latin_keeps_french_part_of_bilingual_names():
    from simulator.ui.theme import latin

    assert latin("Rue d'Alger زنقة الجزائر") == "Rue d'Alger"
    assert latin("Rue sans nom") == "Rue sans nom"
