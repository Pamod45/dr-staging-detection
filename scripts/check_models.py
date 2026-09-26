"""Check models/ against the app config.

    python scripts/check_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src import registry     # noqa: E402
from src.data import STRATEGY_HISTORY  # noqa: E402
from src.decision_trail import TRAIL   # noqa: E402
from src.pages_assets import FINAL_FIGURES  # noqa: E402

problems = registry.validate_config()
print("Config rules:", "OK" if not problems else "")
for p in problems:
    print("  -", p)

print("\nModels:")
for r in registry.check_all():
    flag = "  " if r["status"] == "ok" else "!!"
    print(f"{flag} {r['model']:20s} {r['notebook']}  {r['input']:>4} px  {r['status']:15s}"
          f" {r['weights']:22s} {('missing: ' + r['missing']) if r['missing'] else ''}")

print("\nResults page files:")
for fig in [d.figure for d in TRAIL if d.figure] + [STRATEGY_HISTORY] + list(FINAL_FIGURES.values()):
    print(("   " if fig.exists() else "!! ") + str(fig.relative_to(C.ROOT)))

print("\nShared files:")
for r in registry.check_shared():
    print(("   " if r["exists"] else "!! ") + r["file"])

screening = registry.check_model(C.SCREENING_MODEL_ID)["status"]
print(f"\nScreening model {C.SCREENING_MODEL_ID}: {screening}")
sys.exit(1 if problems or screening != "ok" else 0)
