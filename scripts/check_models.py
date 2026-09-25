"""Check the models/ folder against the app's config, without starting Streamlit.

    python scripts/check_models.py

Exits with code 1 if the config breaks a rule or the screening model is not ready.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src import registry     # noqa: E402

problems = registry.validate_config()
print("Config rules:", "OK" if not problems else "")
for p in problems:
    print("  -", p)

print("\nModels:")
for r in registry.check_all():
    flag = "  " if r["status"] == "ok" else "!!"
    print(f"{flag} {r['model']:20s} {r['notebook']}  {r['input']:>4} px  {r['status']:15s}"
          f" {r['weights']:22s} {('missing: ' + r['missing']) if r['missing'] else ''}")

print("\nResults page figures:")
for title, _, fig in C.RESULTS_BLOCKS:
    if fig is not None:
        print(("   " if fig.exists() else "!! ") + str(fig.relative_to(C.ROOT)))

print("\nShared files:")
for r in registry.check_shared():
    print(("   " if r["exists"] else "!! ") + r["file"])

screening = registry.check_model(C.SCREENING_MODEL_ID)["status"]
print(f"\nScreening model {C.SCREENING_MODEL_ID}: {screening}")
sys.exit(1 if problems or screening != "ok" else 0)
