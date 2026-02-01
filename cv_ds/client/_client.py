import json
import os
from typing import Any, Generator

import httpx

from cv_ds import MDC_API_KEY_NAME

error_api_key_missing_msg = (
    "The API key for Mozilla Data Collective is not set. "
    + "Please provide it as an argument or "
    + f"set the `{MDC_API_KEY_NAME}` environment variable."
)


class BearerAuth(httpx.Auth):
    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, None, None]:
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


class MozillaDataCollectiveClient(httpx.Client):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://datacollective.mozillafoundation.org/api",
        **kwargs: Any,
    ):
        if api_key is None:
            if not (api_key := os.getenv(MDC_API_KEY_NAME)):
                raise ValueError(error_api_key_missing_msg)

        auth = BearerAuth(api_key)

        headers = json.loads(json.dumps(kwargs.pop("headers", None) or {}))

        super().__init__(base_url=base_url, auth=auth, headers=headers, **kwargs)


class MozillaDataCollectiveAsyncClient(httpx.AsyncClient):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://datacollective.mozillafoundation.org/api",
        **kwargs: Any,
    ):
        if api_key is None:
            if not (api_key := os.getenv(MDC_API_KEY_NAME)):
                raise ValueError(error_api_key_missing_msg)

        auth = BearerAuth(api_key)
        headers = json.loads(json.dumps(kwargs.pop("headers", None) or {}))

        super().__init__(base_url=base_url, auth=auth, headers=headers, **kwargs)
