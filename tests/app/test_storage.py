from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from google.cloud import storage
from pytest_mock import MockerFixture

from app.settings import settings
from app.storage import gcs_object_exists, generate_signed_url, get_object_size


@pytest.fixture
def mock_gcs_client(mocker: MockerFixture):
    client = MagicMock(spec=storage.Client)
    mocker.patch("app.storage._gcs_client", return_value=client)
    return client


class TestGcsObjectExists:
    def test_gcs_object_exists(self, mock_gcs_client: MagicMock):
        mock_bucket = mock_gcs_client.bucket.return_value
        mock_blob = mock_bucket.blob.return_value
        mock_blob.exists.return_value = True

        assert gcs_object_exists("test-bucket", "path/to/file.csv") is True

    def test_gcs_object_not_exists(self, mock_gcs_client: MagicMock):
        mock_bucket = mock_gcs_client.bucket.return_value
        mock_blob = mock_bucket.blob.return_value
        mock_blob.exists.return_value = False

        assert gcs_object_exists("test-bucket", "path/to/file.csv") is False


class TestGetObjectSize:
    def test_get_object_size_found(self, mock_gcs_client: MagicMock):
        mock_blob = MagicMock()
        mock_blob.size = 1024

        mock_bucket = mock_gcs_client.bucket.return_value
        mock_bucket.get_blob.return_value = mock_blob

        assert get_object_size("test-bucket", "path/to/file.csv") == 1024

    def test_get_object_size_not_found(self, mock_gcs_client: MagicMock):
        mock_bucket = mock_gcs_client.bucket.return_value
        mock_bucket.get_blob.return_value = None

        assert get_object_size("test-bucket", "path/to/file.csv") is None


class TestGenerateSignedUrl:
    def test_generate_signed_url_with_download_filename(
        self, mock_gcs_client: MagicMock
    ):
        mock_blob = MagicMock()
        mock_blob.generate_signed_url.return_value = (
            "https://storage.example.com/signed"
        )

        mock_bucket = mock_gcs_client.bucket.return_value
        mock_bucket.blob.return_value = mock_blob

        signed_url = generate_signed_url(
            "test-bucket", "path/to/file.csv", download_filename="filename.csv"
        )

        assert signed_url == "https://storage.example.com/signed"
        mock_blob.generate_signed_url.assert_called_once_with(
            expiration=timedelta(seconds=settings.SIGNED_URL_TTL_SECONDS),
            version="v4",
            response_disposition='attachment; filename="filename.csv"',
        )

    def test_generate_signed_url_without_download_filename(
        self, mock_gcs_client: MagicMock
    ):
        mock_blob = MagicMock()
        mock_blob.generate_signed_url.return_value = (
            "https://storage.example.com/signed"
        )

        mock_bucket = mock_gcs_client.bucket.return_value
        mock_bucket.blob.return_value = mock_blob

        signed_url = generate_signed_url("test-bucket", "path/to/file.csv")

        assert signed_url == "https://storage.example.com/signed"
        mock_blob.generate_signed_url.assert_called_once_with(
            expiration=timedelta(seconds=settings.SIGNED_URL_TTL_SECONDS),
            version="v4",
            response_disposition=None,
        )
