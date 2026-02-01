import json
import os
from datetime import datetime
from typing import Any, Generator

import httpx
from pydantic import BaseModel, ConfigDict, Field

from cv_ds import MDC_API_KEY_NAME

error_api_key_missing_msg = (
    "The API key for Mozilla Data Collective is not set. "
    + "Please provide it as an argument or "
    + f"set the `{MDC_API_KEY_NAME}` environment variable."
)


class Organization(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(..., description="The name of the organization")
    slug: str = Field(..., description="The slug of the organization")


class DatasetDetails(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(..., description="The ID of the dataset")
    slug: str = Field(..., description="The slug of the dataset")
    name: str = Field(..., description="The name of the dataset")
    shortDescription: str = Field(
        ..., description="The short description of the dataset"
    )
    longDescription: str = Field(..., description="The long description of the dataset")
    locale: str = Field(..., description="The locale of the dataset")
    sizeBytes: int = Field(..., description="The size of the dataset in bytes")
    createdAt: datetime = Field(..., description="The creation date of the dataset")
    organization: Organization = Field(
        ..., description="The organization of the dataset"
    )
    license: str = Field(..., description="The license of the dataset")
    licenseAbbreviation: str = Field(
        ..., description="The abbreviation of the license of the dataset"
    )
    task: str = Field(..., description="The task of the dataset")
    format: str = Field(..., description="The format of the dataset")
    datasetUrl: str = Field(..., description="The URL of the dataset")


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
        response = self.get(f"/datasets/{dataset_id}")
        response.raise_for_status()
        print()
        print()
        print()
        print(response.json())
        print()
        print()
        print()
        return DatasetDetails.model_validate(response.json())


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
        response = await self.get(f"/datasets/{dataset_id}")
        response.raise_for_status()
        return DatasetDetails.model_validate(response.json())
