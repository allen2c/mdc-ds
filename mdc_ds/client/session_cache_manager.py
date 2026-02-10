"""Session cache management for resumable downloads."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mdc_ds.types.download_session import DownloadSession
from mdc_ds.types.session_cache import SessionCache

logger = logging.getLogger(__name__)


class SessionCacheManager:
    """Manages download session caching and lifecycle.

    Handles session persistence to enable smart resume across download interruptions.
    """

    def __init__(self, cache_dir: Path):
        """Initialize session cache manager.

        Args:
            cache_dir: Directory for storing session cache files
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, dataset_id: str) -> Path:
        """Get the file path for session cache.

        Cache file stored in cache_dir: {dataset_id}.session.json
        """
        return self.cache_dir.joinpath(f"{dataset_id}.session.json")

    def load(self, dataset_id: str) -> SessionCache | None:
        """Load and validate cached session from disk.

        Returns None if cache doesn't exist or fails validation.
        Validates session hasn't expired (11h threshold for 12h validity).
        """
        cache_path = self.get_cache_path(dataset_id)

        if not cache_path.exists():
            logger.debug(f"No session cache found for {dataset_id}")
            return None

        try:
            cache_data: dict[str, Any] = json.loads(cache_path.read_text())
            cache = SessionCache.model_validate(cache_data)

            # Validate expiration with 11h threshold (12h validity - 1h safety margin)
            if datetime.now(timezone.utc) > cache.expires_at:
                logger.info(f"Session cache expired for {dataset_id}")
                return None

            if (datetime.now(timezone.utc) - cache.created_at) > timedelta(hours=11):
                logger.info(
                    f"Session cache too old for {dataset_id}, treating as expired"
                )
                return None

            logger.debug(f"Session cache valid for {dataset_id}")
            return cache

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to load session cache for {dataset_id}: {e}")
            return None

    def save(self, dataset_id: str, session: DownloadSession) -> SessionCache:
        """Save session to cache file and return SessionCache object.

        Stores essential fields and full session backup.
        Uses Pydantic datetime serialization for consistency.
        """
        cache = SessionCache(
            dataset_id=dataset_id,
            download_url=session.downloadUrl,
            total_size_bytes=session.sizeBytes,
            filename=session.filename,
            created_at=datetime.now(timezone.utc),
            expires_at=session.expiresAt,
            checksum=getattr(session, "checksum", None),
            full_session=session.model_dump(),
        )

        cache_path = self.get_cache_path(dataset_id)
        cache_path.write_text(cache.model_dump_json(indent=2))
        logger.debug(f"Session cache saved for {dataset_id}")

        return cache

    def clear(self, dataset_id: str) -> None:
        """Remove session cache file for dataset.

        Called after successful download completion.
        """
        cache_path = self.get_cache_path(dataset_id)
        cache_path.unlink(missing_ok=True)
        logger.debug(f"Session cache cleared for {dataset_id}")
