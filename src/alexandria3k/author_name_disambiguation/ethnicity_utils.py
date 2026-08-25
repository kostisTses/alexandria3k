import subprocess
import time
import os
import csv
import tempfile

CLASSIFIER = os.path.expanduser("~/name-ethnicity-classifier")
MODEL = "28_nationalities_english_once"
TEMP = tempfile.gettempdir()
CHUNK_IN_P = os.path.join(TEMP, "chunk.csv")
CHUNK_OUT_P = os.path.join(TEMP, "chunk_out.csv")
CHUNK_SIZE = 500000


def classify_names(names):
    """
    Runs name-ethnicity-classifier on the given names
    Processes one chunk at a time to prevent out of memory errors
    Predictions keep the order of the names given to the classifier
    """
    start = time.time()
    print("running classifier")
    names = list(names)

    for i in range(0, len(names), CHUNK_SIZE):
        chunk_names = names[i : i + CHUNK_SIZE]

        with open(CHUNK_IN_P, "w", newline="", encoding="utf-8") as chunk:
            writer = csv.writer(chunk)
            writer.writerow(["names"])
            for given, family in chunk_names:
                writer.writerow([f"{given} {family}"])

        classifier = subprocess.run(
            [
                "python3",
                "predict_ethnicity.py",
                "-i",
                CHUNK_IN_P,
                "-o",
                CHUNK_OUT_P,
                "-m",
                MODEL,
                "-d",
                "gpu",
                "-b",
                "1024",
            ],
            cwd=CLASSIFIER,
            check=False,
            stdout=subprocess.DEVNULL,
        )

        if classifier.returncode != 0:
            print(f"\nchunk at {i} failed, skipping")
            continue

        with open(CHUNK_OUT_P, newline="", encoding="utf-8") as chunk_out:
            classifications = list(csv.reader(chunk_out))[1:]

        print(
            f"\r{i + len(chunk_names)}/{len(names)} classified "
            f"{time.time() - start:.2f}s",
            end="",
            flush=True,
        )
        chunk = [
            (given, family, classification[1], float(classification[2]))
            for (given, family), classification in zip(
                chunk_names, classifications
            )
        ]
        yield chunk

    print(f"\n{time.time() - start:.2f}s")