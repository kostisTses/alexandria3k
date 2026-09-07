"""
File contains method that handles classfication of ethnicities.
"""

import time

from n2e import predict_ethnicities

MODEL = "28_nationalities_english_once"
CHUNK_SIZE = 500000
BATCH_SIZE = 1024


def classify_names(names: list):
    """
    Handles classifying names using the n2e classfier.
    Yields one chunk at a time to prevent memory issues

    :param names: the list of names that will get classfied
    :type names: list

    :yield: yields a chunk of (given, family, ethnicity, confidence)

    """
    start = time.time()
    print("running classifier")

    if not isinstance(names, list):
        names = list(names)

    for i in range(0, len(names), CHUNK_SIZE):
        chunk_names = names[i : i + CHUNK_SIZE]

        predictions = predict_ethnicities(
            [f"{given} {family}" for given, family in chunk_names],
            batch_size=BATCH_SIZE,
            model=MODEL,
        )

        print(
            f"\r{i + len(chunk_names)}/{len(names)} classified "
            f"{time.time() - start:.2f}s",
            end="",
            flush=True,
        )
        chunk = [
            (given, family, ethnicity, confidence)
            for (given, family), (ethnicity, confidence) in zip(
                chunk_names, predictions
            )
        ]
        yield chunk

    print(f"\n{time.time() - start:.2f}s")
