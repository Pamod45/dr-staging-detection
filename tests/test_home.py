"""Home page: dr_facts.md is parsed into sections and grade definitions, the drawings show the
right signs, and the page renders."""
from pathlib import Path

import cv2
import pytest
from streamlit.testing.v1 import AppTest

from src import config as C
from src import content, lesions
from tests.helpers import fundus

MAIN = str(Path(__file__).resolve().parents[1] / "app" / "main.py")
SAMPLE = Path(__file__).resolve().parent / "data" / "dr_facts_sample.md"


@pytest.fixture
def facts(monkeypatch):
    monkeypatch.setattr(content, "FACTS", SAMPLE)
    return content.sections()


def test_sections_split_on_headings(facts):
    assert list(facts)[:3] == ["What diabetic retinopathy is", "The five ICDR grades",
                               "Referable or not"]
    assert "<!--" not in " ".join(facts.values())


def test_grade_definitions_join_wrapped_lines(facts):
    defs = content.grade_definitions(facts["The five ICDR grades"])
    assert sorted(defs) == [0, 1, 2, 3, 4]
    assert defs[1][0] == "Mild" and "small red dots" in defs[1][1]
    assert "one or more quadrant" in defs[3][1]
    assert "Wilkinson" in content.after_bullets(facts["The five ICDR grades"])


def make_lesion_set(folder, stem="IDRiD_99"):
    """Synthetic photo plus 1-bit TIFF masks named like the IDRiD segmentation release."""
    import numpy as np
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=True)
    h, w = 600, 900
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (450, 300), 290, (40, 90, 190), -1)
    masks = {k: np.zeros((h, w), np.uint8) for k in ("MA", "HE", "EX", "SE", "OD")}
    for x, y in [(300, 200), (350, 380), (420, 260), (520, 420)]:
        cv2.circle(masks["MA"], (x, y), 2, 255, -1)
    cv2.circle(masks["HE"], (330, 300), 14, 255, -1)
    cv2.circle(masks["HE"], (560, 200), 9, 255, -1)
    cv2.circle(masks["EX"], (400, 420), 6, 255, -1)
    cv2.ellipse(masks["SE"], (380, 180), (16, 10), 0, 0, 360, 255, -1)
    cv2.circle(masks["OD"], (600, 290), 50, 255, -1)
    cv2.imwrite(str(folder / f"{stem}.jpg"), img)
    for k, m in masks.items():
        Image.fromarray(m > 0).convert("1").save(folder / f"{stem}_{k}.tif", compression="group4")


def test_lesion_set_is_found_and_counted(tmp_path, monkeypatch):
    make_lesion_set(tmp_path)
    monkeypatch.setattr(lesions, "LESION_DIR", tmp_path)
    a = lesions.find()
    assert a.name == "IDRiD_99" and set(a.masks) == {"MA", "HE", "EX", "SE", "OD"}
    assert lesions.counts(a) == {"MA": 4, "HE": 2, "EX": 1, "SE": 1}
    assert lesions.example(a, "HE") == pytest.approx((330, 300), abs=1)


def test_callout_points_at_every_shown_sign(tmp_path, monkeypatch):
    make_lesion_set(tmp_path)
    monkeypatch.setattr(lesions, "LESION_DIR", tmp_path)
    a = lesions.find()
    svg, width = lesions.callout_svg(a, ("MA", "HE", "EX", "SE", "OD"))
    assert svg.count("marker-end=") == 5 and "first appears at grade 1 (Mild)" in svg
    svg_two, _ = lesions.callout_svg(a, ("HE",))
    assert svg_two.count("marker-end=") == 1 and width > lesions.FIG_W
    assert lesions.crop(a, "SE").shape == (300, 300, 3)


def test_sets_are_found_per_grade(tmp_path, monkeypatch):
    make_lesion_set(tmp_path / "grade_2", "IDRiD_61")
    make_lesion_set(tmp_path / "grade_4", "IDRiD_17")
    make_lesion_set(tmp_path, "IDRiD_17")
    monkeypatch.setattr(lesions, "LESION_DIR", tmp_path)
    sets = lesions.by_grade()
    assert {g: a.name for g, a in sets.items()} == {2: "IDRiD_61", 4: "IDRiD_17"}


def test_missing_mask_means_no_lesion_set(tmp_path, monkeypatch):
    make_lesion_set(tmp_path)
    (tmp_path / "IDRiD_99_SE.tif").unlink()
    monkeypatch.setattr(lesions, "LESION_DIR", tmp_path)
    assert lesions.find() is None


def test_home_renders(tmp_path, monkeypatch):
    monkeypatch.setattr(content, "FACTS", SAMPLE)
    grades = tmp_path / "grades"
    grades.mkdir()
    for g in range(5):
        cv2.imwrite(str(grades / f"grade_{g}.jpg"), fundus(300, 300, 140, notch=False))
    monkeypatch.setattr(C, "SAMPLES_DIR", tmp_path)
    split = tmp_path / "split.csv"
    split.write_text("id_code,diagnosis,split\n" + "\n".join(
        f"x{i},{i % 5},{('train', 'val', 'test')[i % 3]}" for i in range(60)))
    monkeypatch.setattr(C, "SPLIT_FILES", {**C.SPLIT_FILES, "v2": split})
    for g, stem in ((2, "IDRiD_61"), (3, "IDRiD_25"), (4, "IDRiD_17")):
        make_lesion_set(tmp_path / "lesions" / f"grade_{g}", stem)
    monkeypatch.setattr(lesions, "LESION_DIR", tmp_path / "lesions")
    at = AppTest.from_file(MAIN, default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5 and len(at.image) >= 5
    captions = " ".join(c.value for c in at.caption)
    assert captions.count("CC BY 4.0") == 3
    assert "IDRiD_17, graded Proliferative DR" in captions
    assert "taken from the annotated Moderate photograph (IDRiD_61)" in captions
    assert [h.value for h in at.header][:3] == ["The five ICDR grades", "Referable or not",
                                                "The datasets"]


def test_home_without_facts_file(tmp_path, monkeypatch):
    monkeypatch.setattr(content, "FACTS", tmp_path / "missing.md")
    at = AppTest.from_file(MAIN, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("could not be found" in e.value for e in at.error)
