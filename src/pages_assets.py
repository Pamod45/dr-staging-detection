"""Exported notebook figures that pages show as images."""
from src import config as C

_RUN04 = C.FIGURES_V2 / "04_full_dense_none_focal_768"

FINAL_FIGURES = {
    "idrid_confusion": _RUN04 / "cm_idrid.png",
}
