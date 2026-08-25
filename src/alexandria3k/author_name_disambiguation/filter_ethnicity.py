"""
Script to filter and populate database with authors from a specific ethnicity
Currently name-ethinicity-classifier is being used and needs to be installed
for it to work
https://github.com/name-ethnicity-classifier
Note that this specific classifier works only with .csv files

Script takes arguments <ethnicity> <path to compressed files> [--rebuild]
 - "ethnicity" is the particular ethnicity filtered, provided by the model
 - "path to compressed files" is the path consisting the compressed jsonl
   files you want to filter
 - "--rebuild" drops classifications.db so every name is classified again

You dont need to populate the database into tables to use this script,
works on compressed files to save disk space

Classifications are kept in classifications.db which holds every name that
has been classified and every compressed file that has been read, so adding
files to the dataset only classifies the names that are new

Script calls a3k --attach-databases so filtered database can be populated
and used
"""

import argparse
import subprocess
import gzip
import json
import os
import sys

import apsw

from alexandria3k.author_name_disambiguation.ethnicity_utils import (
    CLASSIFIER,
    MODEL,
    classify_names,
)

WORK_DIR = "alexandria3k"
CLASSIFICATIONS_DB = "classifications.db"
CONFIDENCE = 95


def create_databases(work_dir):
    """Opens classifications.db in work_dir, creating its tables when missing"""
    os.makedirs(work_dir, exist_ok=True)
    database = apsw.Connection(os.path.join(work_dir, CLASSIFICATIONS_DB))
    database.execute(
        "CREATE TABLE IF NOT EXISTS classified_names "
        "(given, family, ethnicity, confidence)"
    )
    database.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_classified_name "
        "ON classified_names(given, family)"
    )
    database.execute("CREATE TABLE IF NOT EXISTS processed_files (file_name)")
    return database


def processed_files(database):
    """Returns the names of the compressed files already read"""
    return {
        row[0]
        for row in database.execute("SELECT file_name FROM processed_files")
    }


def split_files(ethnicity, ethnicity_names, path, work_dir):
    """Copies works with an author of the ethnicity into their own files"""
    names = set(ethnicity_names)
    output_p = os.path.join(work_dir, f"{ethnicity}_files")
    os.makedirs(output_p, exist_ok=True)

    files = [f.name for f in os.scandir(path) if f.name.endswith(".jsonl.gz")]

    for i, file in enumerate(files):
        with (
            gzip.open(f"{path}/{file}", "rt", encoding="utf-8") as f,
            gzip.open(
                f"{output_p}/{ethnicity}{i}.jsonl.gz", "wt", encoding="utf-8"
            ) as out,
        ):
            for jsonl in f:
                work = json.loads(jsonl)
                for author in work.get("author", []):
                    given = author.get("given")
                    family = author.get("family")
                    if (given, family) in names:
                        out.write(jsonl)
                        break
        print(f"\r{i + 1}/{len(files)} files filtered", end="", flush=True)
    return output_p


def extract_names(path, database):
    """
    Extracts author names from the compressed files that have not been read
    Returns the names and the files they were read from
    """
    names = set()
    files = [
        file
        for file in os.scandir(path)
        if file.name.endswith(".jsonl.gz")
        and file.name not in processed_files(database)
    ]

    read_files = []
    for i, file in enumerate(files, start=1):
        try:
            with gzip.open(file.path, "rt", encoding="utf-8") as f:
                for jsonl in f:
                    work = json.loads(jsonl)
                    for author in work.get("author", []):
                        given = author.get("given")
                        family = author.get("family")
                        if not given or not family:
                            continue
                        names.add((given, family))
        except (gzip.BadGzipFile, EOFError) as error:
            print(f"\nskipping {file.name}: {error}")
            continue

        read_files.append(file.name)
        print(f"\r{i}/{len(files)} files loaded", end="", flush=True)

    print("\nextracted names")
    return names, read_files


def unclassified(database, names):
    """Returns the names that are not in classified_names yet"""
    return [
        (given, family)
        for given, family in names
        if not list(
            database.execute(
                "SELECT 1 FROM classified_names WHERE given = ? AND family = ?",
                (given, family),
            )
        )
    ]


def mark_processed(database, files):
    """Stores the compressed files whose names have been classified"""
    with database:
        for name in files:
            database.execute("INSERT INTO processed_files VALUES (?)", (name,))


def filter_names(database, ethnicity):
    """Returns the names classified as the given ethnicity"""
    return list(
        database.execute(
            "SELECT given, family FROM classified_names "
            "WHERE ethnicity = ? AND confidence >= ?",
            (ethnicity, CONFIDENCE),
        )
    )


def populate_database(ethnicity, path, classifications_db):
    """Attaches the database and populates it with works of the ethnicity"""
    subprocess.run(
        [
            "a3k",
            "populate",
            f"{ethnicity}.db",
            "crossref",
            path,
            "--attach-databases",
            f"attached:{classifications_db}",
            "--row-selection",
            "EXISTS (SELECT 1 FROM attached.classified_names "
            "WHERE classified_names.given IS work_authors.given "
            "AND classified_names.family IS work_authors.family "
            f"AND classified_names.ethnicity = '{ethnicity}' "
            f"AND classified_names.confidence >= {CONFIDENCE})",
        ],
        check=True,
    )


def main():
    """
    Takes arguments <ethnicity> <path/to/compressed/files> [--rebuild]
    Classifier used is name-ethnicity-classifier
    Names and their predicted ethnicity are stored in classifications.db
    together with the compressed files they were read from
    --rebuild drops classifications.db so every name is classified again
    Attach and populate the database using a3k --attach-databases
    Script produces populated tables which can be used for other processes
    """
    parser = argparse.ArgumentParser(
        description="Filter and populate a database "
        "with authors of one ethnicity"
    )
    parser.add_argument(
        "ethnicity", help="ethnicity to filter, as named by the model"
    )
    parser.add_argument(
        "path", help="directory containing the compressed jsonl files"
    )
    parser.add_argument(
        "--work-dir",
        default=WORK_DIR,
        help="where classifications.db and the filtered files are kept",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop classifications.db so every name is classified again",
    )
    arguments = parser.parse_args()
    ethnicity, path = arguments.ethnicity, arguments.path

    with open(
        f"{CLASSIFIER}/model_configurations/{MODEL}/nationalities.json",
        encoding="utf-8",
    ) as f:
        nationalities = json.load(f)

    if ethnicity not in nationalities:
        sys.exit(
            f"unknown ethnicity '{ethnicity}'\n"
            f"supported: {', '.join(nationalities)}"
        )

    classifications_db = os.path.join(arguments.work_dir, CLASSIFICATIONS_DB)
    if arguments.rebuild and os.path.exists(classifications_db):
        os.remove(classifications_db)

    database = create_databases(arguments.work_dir)

    names, read_files = extract_names(path, database)
    names = unclassified(database, names)

    for chunk in classify_names(names):
        with database:
            for row in chunk:
                database.execute(
                    "INSERT OR IGNORE INTO classified_names VALUES (?, ?, ?, ?)",
                    row,
                )

    mark_processed(database, read_files)

    ethnicity_names = filter_names(database, ethnicity)
    print(f"{len(ethnicity_names)} {ethnicity} names")

    output_p = split_files(ethnicity, ethnicity_names, path, arguments.work_dir)
    print("")
    print("populating names using a3k")
    populate_database(ethnicity, output_p, classifications_db)


if __name__ == "__main__":
    main()
