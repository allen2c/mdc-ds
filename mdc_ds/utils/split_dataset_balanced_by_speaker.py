import logging

from datasets import Dataset, DatasetDict

# Configure logger
logger = logging.getLogger(__name__)


def split_dataset_balanced_by_speaker(
    dataset: Dataset, ratios: tuple = (0.8, 0.1, 0.1), client_col: str = "client_id"
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Splits a Hugging Face Dataset into Train/Dev/Test sets while ensuring:
    1. Speaker Isolation: No client_id overlaps between splits.
    2. Balanced Sample Counts: Uses a greedy algorithm to approximate the target ratios based on sample counts, not just speaker counts.

    Args:
        dataset (Dataset): The input Hugging Face dataset.
        ratios (tuple): Target ratios for (Train, Dev, Test). Sum must be 1.0.
        client_col (str): The column name representing the speaker ID.

    Returns:
        DatasetDict: A dictionary containing 'train', 'dev', and 'test' splits.
    """  # noqa: E501

    # 1. Basic validation
    if not sum(ratios) == 1.0:
        raise ValueError("Sum of ratios must be 1.0 (e.g., 0.8, 0.1, 0.1)")

    logger.info("Starting greedy speaker split...")

    # 2. Count samples per speaker
    # Converting to Pandas is usually fast enough for metadata analysis
    df = dataset.select_columns([client_col]).to_pandas()
    speaker_counts = df[client_col].value_counts().to_dict()  # type: ignore

    # 3. Sort speakers by contribution (Highest -> Lowest)
    # This is critical for the greedy "water-filling" strategy to minimize variance.
    sorted_speakers = sorted(
        speaker_counts.keys(), key=lambda k: speaker_counts[k], reverse=True
    )

    # 4. Calculate target sample counts
    total_samples = len(dataset)
    target_train = total_samples * ratios[0]
    target_dev = total_samples * ratios[1]
    target_test = total_samples * ratios[2]

    # 5. Initialize containers
    train_ids, dev_ids, test_ids = set(), set(), set()
    count_train, count_dev, count_test = 0, 0, 0

    # 6. Greedy Allocation Loop
    for speaker in sorted_speakers:
        n_samples = speaker_counts[speaker]

        # Calculate the "gap" (how far we are from the target)
        # We assign the current speaker to the bucket with the largest remaining need.
        gap_train = target_train - count_train
        gap_dev = target_dev - count_dev
        gap_test = target_test - count_test

        # Determine winner
        if gap_train >= gap_dev and gap_train >= gap_test:
            train_ids.add(speaker)
            count_train += n_samples
        elif gap_dev >= gap_test:
            dev_ids.add(speaker)
            count_dev += n_samples
        else:
            test_ids.add(speaker)
            count_test += n_samples

    # 7. Apply filters to create actual datasets
    # Note: Using sets for O(1) lookups during filtering
    logger.info("Filtering dataset into splits...")

    final_splits = DatasetDict(
        {
            "train": dataset.filter(lambda x: x[client_col] in train_ids),
            "dev": dataset.filter(lambda x: x[client_col] in dev_ids),
            "test": dataset.filter(lambda x: x[client_col] in test_ids),
        }
    )

    # 8. Log Final Statistics
    logger.info("-" * 40)
    logger.info(f"Total Samples: {total_samples}")
    logger.info(
        f"Train: {len(final_splits['train'])} ({len(final_splits['train']) / total_samples:.2%}) | Target: {ratios[0]:.0%}"  # noqa: E501
    )
    logger.info(
        f"Dev:   {len(final_splits['dev'])} ({len(final_splits['dev']) / total_samples:.2%}) | Target: {ratios[1]:.0%}"  # noqa: E501
    )
    logger.info(
        f"Test:  {len(final_splits['test'])} ({len(final_splits['test']) / total_samples:.2%}) | Target: {ratios[2]:.0%}"  # noqa: E501
    )
    logger.info("-" * 40)

    return final_splits["train"], final_splits["dev"], final_splits["test"]
