from src import config as C

_RUN04 = C.FIGURES_V2 / "04_full_dense_none_focal_768"

FINAL_FIGURES = {
    "idrid_confusion": _RUN04 / "cm_idrid.png",
}

# Hand-drawn pipeline diagrams, dark versions of the report figures, 1600 px wide viewBoxes.
DIAGRAMS = {name: C.CONTENT_DIR / "diagrams" / f"{name}.svg"
            for name in ("preprocessing", "architecture", "training")}
DIAGRAM_WIDTH = 1600
