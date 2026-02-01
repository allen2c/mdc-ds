import json
import logging
import os
from typing import Any, Generator

import httpx

from cv_ds import MDC_API_KEY_NAME
from cv_ds.types.dataset_details import DatasetDetails
from cv_ds.types.download_session import DownloadSession

logger = logging.getLogger(__name__)

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

    def get_dataset_details(self, dataset_id: str) -> DatasetDetails:
        """Retrieves the details of a specific dataset."""

        from cv_ds.registry import get_dataset_details

        if cached_ds := get_dataset_details(dataset_id):
            logger.debug(f"Dataset input {dataset_id} found in cache")
            return cached_ds

        response = self.get(f"/datasets/{dataset_id}")
        response.raise_for_status()
        return DatasetDetails.model_validate(response.json())

    def get_dataset_download_session(self, dataset_id: str) -> DownloadSession:
        """Creates a download session and returns the dataset's download URL for direct download from storage. The user must have previously agreed to the dataset's terms of use through the web interface."""  # noqa: E501

        from cv_ds.registry import get_dataset_details

        if not (ds := get_dataset_details(dataset_id)):
            ds = self.get_dataset_details(dataset_id)

        response = self.post(f"/datasets/{ds.id}/download")
        response.raise_for_status()
        return DownloadSession.model_validate(response.json())


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

    async def get_dataset_details(self, dataset_id: str) -> DatasetDetails:
        """Retrieves the details of a specific dataset."""

        from cv_ds.registry import get_dataset_details

        if cached_ds := get_dataset_details(dataset_id):
            logger.debug(f"Dataset input {dataset_id} found in cache")
            return cached_ds

        response = await self.get(f"/datasets/{dataset_id}")
        response.raise_for_status()
        return DatasetDetails.model_validate(response.json())

    async def get_dataset_download_session(self, dataset_id: str) -> DownloadSession:
        """Creates a download session and returns the dataset's download URL for direct download from storage. The user must have previously agreed to the dataset's terms of use through the web interface."""  # noqa: E501

        from cv_ds.registry import get_dataset_details

        if not (ds := get_dataset_details(dataset_id)):
            ds = await self.get_dataset_details(dataset_id)

        response = await self.post(f"/datasets/{ds.id}/download")
        response.raise_for_status()
        return DownloadSession.model_validate(await response.json())
