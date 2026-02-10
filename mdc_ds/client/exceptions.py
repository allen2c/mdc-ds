"""Custom exceptions for MDC client operations."""


class SessionExpiredError(Exception):
    """Raised when download session/token has expired (HTTP 403).

    Indicates need to refresh session from API.
    """

    def __init__(self, message: str = "Download session expired, refresh required"):
        self.message = message
        super().__init__(self.message)
