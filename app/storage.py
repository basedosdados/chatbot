from datetime import timedelta
from functools import cache

from google.cloud import storage

from app.settings import settings


@cache
def _gcs_client() -> storage.Client:  # pragma: no cover
    return storage.Client(
        project=settings.GOOGLE_BIGQUERY_PROJECT,
        credentials=settings.GOOGLE_CREDENTIALS,
    )


def gcs_object_exists(bucket: str, object_key: str) -> bool:
    """Check whether a GCS object exists.

    Args:
        bucket (str): The name of the GCS bucket.
        object_key (str): The object key within the bucket.

    Returns:
        bool: True if the object exists. False otherwise.
    """
    blob = _gcs_client().bucket(bucket).blob(object_key)
    return blob.exists()


def get_object_size(bucket: str, object_key: str) -> int | None:
    """Get the size in bytes of a GCS object.

    Args:
        bucket (str): The name of the GCS bucket.
        object_key (str): The object key within the bucket.

    Returns:
        int | None: The size in bytes if the object was found. None otherwise.
    """
    blob = _gcs_client().bucket(bucket).get_blob(object_key)
    return blob.size if blob is not None else None


def generate_signed_url(
    bucket: str, object_key: str, download_filename: str | None = None
) -> str:
    """Generate a short-lived v4 signed URL for a GCS object.

    Args:
        bucket (str): The name of the GCS bucket.
        object_key (str): The object key within the bucket.
        download_filename (str | None, optional): If set, forces the browser
            to save the object under this name via `Content-Disposition`.
            Defaults to None.

    Returns:
        str: A signed download URL valid for `SIGNED_URL_TTL_SECONDS` seconds.
    """
    blob = _gcs_client().bucket(bucket).blob(object_key)

    response_disposition = (
        f'attachment; filename="{download_filename}"' if download_filename else None
    )

    return blob.generate_signed_url(
        expiration=timedelta(seconds=settings.SIGNED_URL_TTL_SECONDS),
        version="v4",
        response_disposition=response_disposition,
    )
