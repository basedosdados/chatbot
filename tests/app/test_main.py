from fastapi import status
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_redirect_root_to_docs():
    response = client.get("/", follow_redirects=False)

    assert response.status_code == status.HTTP_307_TEMPORARY_REDIRECT
    assert response.headers["location"] == "/docs"


def test_health_check():
    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "healthy"}
