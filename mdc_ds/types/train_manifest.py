from google_language_support import LanguageCodes
from pydantic import BaseModel, Field


class TrainManifest(BaseModel):
    audio_path: str = Field(..., description="The path to the audio file")
    text: str = Field(..., description="The text of the audio file")
    language: LanguageCodes | str | None = Field(
        default=None, description="The language of the audio file"
    )
