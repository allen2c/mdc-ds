import io
import logging
import os
import tarfile
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

import pandas as pd
from datasets import Dataset, DatasetDict
from google_language_support import LanguageCodes
from pydub import AudioSegment

from mdc_ds.types.feature import feature
from mdc_ds.utils.split_dataset_balanced_by_speaker import (
    split_dataset_balanced_by_speaker,
)

logger = logging.getLogger(__name__)

slug_name = "common-voice-v24-english-en-au-subset-fo-0447e8a6"


class AudioProcessor:
    """
    Helper class to handle tarfile opening within worker processes.
    Ensures that the tar file is opened once per process to avoid overhead.
    """

    def __init__(self, tar_path: str):
        self.tar_path = tar_path
        # The tar handle will be initialized lazily in the worker process
        self._tar: Optional[tarfile.TarFile] = None

    def __call__(self, example):
        if self._tar is None:
            self._tar = tarfile.open(self.tar_path, "r:gz")

        audio_path = example["audio_path"]

        # Extract file from tar
        audio_tar_obj = self._tar.extractfile(audio_path)
        if not audio_tar_obj:
            raise ValueError(f"Audio tar object not found: {audio_path}")

        # Audio processing logic (same as original)
        audio_seg: AudioSegment = AudioSegment.from_file(
            io.BytesIO(audio_tar_obj.read())
        )
        audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)
        mp3_io = io.BytesIO()
        audio_seg.export(mp3_io, format="mp3", bitrate="128k")
        audio_bytes = mp3_io.getvalue()

        # Return the processed audio bytes
        return {"audio": audio_bytes}

    def __del__(self):
        if self._tar:
            self._tar.close()


class TrainManifest(TypedDict):
    client_id: str
    audio_path: str
    text: str
    language: LanguageCodes


def get_metadata(downloaded_filepath: Path | str) -> List[TrainManifest]:
    tar_root = Path("commonvoice-v24_en-AU")
    train_manifests: List[TrainManifest] = []

    # 1. Parse TSV files to build metadata lists (Lightweight)
    # We open tar only to get the TSV files initially
    logger.debug("Parsing CSV files to build metadata lists")
    with tarfile.open(downloaded_filepath, "r:gz") as tar:
        might_manifest_filepaths = [
            m.name
            for m in tar.getmembers()
            if m.name.lower().endswith((".csv", ".json", ".tsv"))
        ]
        try:
            train_tar_filepath = tar.extractfile(
                str(tar_root.joinpath("commonvoice-v24_en-AU.csv"))
            )
            if not train_tar_filepath:
                raise ValueError(
                    "Train tar file not found, "
                    f"manifest files might be in {might_manifest_filepaths}"
                )
        except KeyError as e:
            logger.error(
                f"Train tar file not found: {e}, "
                f"manifest files might be in {might_manifest_filepaths}"
            )
            raise e

        train_df = pd.read_csv(train_tar_filepath)

        # Convert to list of dicts immediately for Dataset.from_list
        for train_row in train_df.itertuples(index=False):
            full_audio_path = str(
                tar_root.joinpath("audio_files").joinpath(train_row.path)  # type: ignore  # noqa: E501
            )
            train_manifests.append(
                {
                    "client_id": train_row.client_id,  # type: ignore
                    "audio_path": full_audio_path,
                    "text": train_row.sentence,  # type: ignore
                    "language": LanguageCodes.ENGLISH,
                }
            )

    return train_manifests


def get_dataset(
    name: Literal["common-voice-v24-english-en-au-subset-fo-0447e8a6"],
    split: Literal["train", "test", "validation"] = "train",
) -> "Dataset":
    from mdc_ds import DEFAULT_MDC_DATASETS_CACHE, MDC_DATASETS_CACHE_NAME
    from mdc_ds.client import MozillaDataCollectiveClient

    cache_path = Path(
        os.getenv(MDC_DATASETS_CACHE_NAME, None) or DEFAULT_MDC_DATASETS_CACHE
    ).joinpath(slug_name)
    cache_path.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and cache_path.is_dir() and any(cache_path.glob("*.json")):
        return DatasetDict.load_from_disk(str(cache_path))[split]

    client = MozillaDataCollectiveClient()
    ds_details = client.get_dataset_details(slug_name)
    downloaded_filepath = client.download_dataset(ds_details.id)

    # 1. Get metadata
    train_manifests = get_metadata(downloaded_filepath)

    # 2. Create initial Datasets (Metadata only, no heavy processing yet)
    # Using from_list is faster than generator for in-memory data
    all_dataset = Dataset.from_list(train_manifests)  # type: ignore

    # Split speakers into train, dev, and test sets in 90/5/5 ratio
    train_ds, dev_ds, test_ds = split_dataset_balanced_by_speaker(
        all_dataset, ratios=(0.9, 0.05, 0.05), client_col="client_id"
    )

    # 3. Apply Audio Processing in Parallel using .map()
    # Initialize the processor with the path to the tar file
    audio_processor = AudioProcessor(str(downloaded_filepath))

    logger.debug("Processing train dataset audio in parallel...")
    train_ds = train_ds.map(
        audio_processor,
        num_proc=4,
        remove_columns=[
            "client_id",
            "audio_path",
        ],  # We don't need the path anymore after extraction
        desc="Decoding train audio",
    )

    logger.debug("Processing dev dataset audio in parallel...")
    dev_ds = dev_ds.map(
        audio_processor,
        num_proc=4,
        remove_columns=["client_id", "audio_path"],
        desc="Decoding dev audio",
    )

    logger.debug("Processing test dataset audio in parallel...")
    test_ds = test_ds.map(
        audio_processor,
        num_proc=4,
        remove_columns=["client_id", "audio_path"],
        desc="Decoding test audio",
    )

    # 4. Cast to target features (Optional but recommended to match original intent)
    # This ensures the 'audio' column matches the type defined in 'feature'
    train_ds = train_ds.cast(feature)
    dev_ds = dev_ds.cast(feature)
    test_ds = test_ds.cast(feature)

    dataset_dict = DatasetDict(
        {
            "train": train_ds,
            "dev": dev_ds,
            "test": test_ds,
        }
    )

    dataset_dict.save_to_disk(str(cache_path))
    return DatasetDict.load_from_disk(str(cache_path))[split]
