from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SessionCache(BaseModel):
    """Cache model for download session persistence.

    Stores session info to enable smart resume across restarts.
    Designed to be resilient to API schema changes.
    """

    model_config: ConfigDict = ConfigDict(extra="allow")

    # Core fields required for resumption (unlikely to change)
    version: str = Field(default="1.0", description="Cache format version")
    dataset_id: str = Field(..., description="Dataset identifier")
    download_url: str = Field(..., description="Signed download URL")
    total_size_bytes: int = Field(..., description="Total file size in bytes")
    filename: str = Field(..., description="Target filename")

    # Timestamp for expiration validation (11h threshold = 12h validity - 1h safety margin)  # noqa: E501
    created_at: datetime = Field(..., description="Cache creation timestamp")
    expires_at: datetime = Field(..., description="Session expiration timestamp")

    # Optional: Additional metadata for integrity verification
    checksum: str | None = Field(default=None, description="File checksum if available")

    # Backup: Full session dict in case API schema changes
    full_session: dict[str, Any] = Field(
        default_factory=dict, description="Full DownloadSession as backup"
    )
