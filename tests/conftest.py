"""Default tests are provider-free and network-denied."""

import os

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        for layer in ("unit", "api", "integration"):
            if f"/{layer}/" in str(item.path):
                item.add_marker(getattr(pytest.mark, layer))


@pytest.fixture(autouse=True)
def no_live_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("OPENAI_API_KEY", "").startswith("sk-"):
        pytest.fail("Test process received a live-looking provider key; use isolated test settings")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-never-a-real-key")
