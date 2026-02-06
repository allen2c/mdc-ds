from typing import TypedDict

from google_language_support import LanguageCodes


class ManifestItem(TypedDict):
    audio_path: str
    text: str
    language: LanguageCodes


class ManifestItemWithClientId(TypedDict):
    client_id: str
