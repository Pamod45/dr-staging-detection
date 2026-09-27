"""Low-memory mode for hosted runs."""
from src import engines


def test_low_memory_setting(monkeypatch):
    monkeypatch.setenv("LOW_MEMORY", "1")
    assert engines.low_memory()
    monkeypatch.setenv("LOW_MEMORY", "0")
    assert not engines.low_memory()


def test_release_clears_the_comparison_cache(monkeypatch):
    cleared = []
    monkeypatch.setattr(engines.comparison_runner, "clear", lambda: cleared.append(True))
    engines.release_comparison_models()
    assert cleared == [True]
