# conftest.py
import pytest

@pytest.fixture(autouse=True)
def mock_environment(monkeypatch):
    """Mock Kubernetes cluster access for all tests"""
    monkeypatch.setenv("KUBECONFIG", "/dev/null")