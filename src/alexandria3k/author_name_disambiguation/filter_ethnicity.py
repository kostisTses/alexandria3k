"""
Script to filter and populate database with authors from a specific ethnicity,
works on jsonl.gzip files

Args: <ethnicity> <path to compressed files>
      [--work-dir DIR] [--output DB] [--rebuild]
 - "ethnicity" is the particular ethnicity filtered, provided by the model
 - "path to compressed files" is the path consisting the compressed jsonl
   files you want to filter
 - "--work-dir" is where classifications.db and the filtered files are kept,
   defaults to $XDG_CACHE_HOME/alexandria3k, or ~/.cache/alexandria3k
 - "--output" is the name of the database to populate,
   defaults to <ethnicity>.db in the current directory
 - "--rebuild" drops classifications.db so every name is classified again

You dont need to populate the database into tables to use this script,
works on compressed files to save disk space

Classifications are kept in classifications.db which holds every name that
has been classified and every compressed file that has been read, so adding
files to the dataset only classifies the names that are new.

Script produces a directory named <ethnicity>_files and contains compressed jsonl files
containing only the works with authors of the specific matching ethnicity.
This helps for the a3k populate part so to scan only the needed entries.

classifications.db and <ethnicity>_files are both stored in working_dir
(defaults to ~/.cache/alexandria3k)

Script calls a3k --attach-databases so filtered database can be populated
and used.

Outputs a database named with --output (defaults to <ethnicity>.db) in current directory.
Schema is the a3k default schema of "a3k populate crossref"
"""

import argparse
import subprocess
import gzip
import json
import os
from multiprocessing import Pool

import apsw

from alexandria3k.author_name_disambiguation.classify import (
    classify_names,
)

WORK_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
    "alexandria3k",
)
CLASSIFICATIONS_DB = "classifications.db"
CONFIDENCE = 95


def create_tables(work_dir):
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


def processed_files(database) -> list:
    """Returns the names of the compressed files already read"""
    return {
        row[0]
        for row in database.execute("SELECT file_name FROM processed_files")
    }


# pylint: disable-next=too-many-locals
def split_files(ethnicity, ethnicity_names, path, work_dir):
    """Copies works with an author of the ethnicity into their own files"""
    names = set(ethnicity_names)
    output_p = os.path.join(work_dir, f"{ethnicity}_files")
    os.makedirs(output_p, exist_ok=True)

    json_files = [
        f.name for f in os.scandir(path) if f.name.endswith(".jsonl.gz")
    ]

    for i, file in enumerate(json_files):
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
        print(
            f"\r{i + 1}/{len(json_files)} files filtered", end="", flush=True
        )
    return output_p


def read_file_names(file_path):
    names = set()
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                work = json.loads(line)
                for author in work.get("author", []):
                    given = author.get("given")
                    family = author.get("family")
                    if not given or not family:
                        continue
                    names.add((given, family))
    except (gzip.BadGzipFile, EOFError) as error:
        return os.path.basename(file_path), None, str(error)
    return os.path.basename(file_path), names, None


def extract_names(path, database):
    processed = processed_files(database)
    jsonl_files = [
        file.path
        for file in os.scandir(path)
        if file.name.endswith(".jsonl.gz") and file.name not in processed
    ]

    names = set()
    read_files = []
    with Pool() as pool:
        for i, (file_name, file_names, error) in enumerate(
            pool.imap_unordered(read_file_names, jsonl_files), start=1
        ):
            if error:
                print(f"\nskipping {file_name}: {error}")
                continue
            names |= file_names
            read_files.append(file_name)
            print(
                f"\r{i}/{len(jsonl_files)} files loaded", end="", flush=True
            )

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


def populate_database(ethnicity, path, classifications_db, output):
    """Attaches the database and populates it with works of the ethnicity"""
    subprocess.run(
        [
            "a3k",
            "populate",
            output,
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


def get_flags():
    """
    Handles console arguments

    :return ethnicity str:  ethnicity to filter
    :return path str:     directory containing the compressed jsonl files
    :return work_dir str: directory holding classifications.db and the filtered files
    :return output str:   file path of the database to populate
    :return classifications_db str:  file path of classifications.db
    :return bool: whether classifications.db is dropped before classifying
    """
    parser = argparse.ArgumentParser(
        description="Filter and populate a database "
        "with authors of one ethnicity"
    )
    parser.add_argument("ethnicity", help="ethnicity to filter")
    parser.add_argument(
        "path", help="directory containing the compressed jsonl files"
    )
    parser.add_argument(
        "--work-dir",
        default=WORK_DIR,
        help="where classifications.db and the filtered files are kept",
    )
    parser.add_argument(
        "--output",
        help="file path of the database to populate, "
        "defaults to <ethnicity>.db in the current directory",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop classifications.db so every name is classified again",
    )
    arguments = parser.parse_args()

    output = arguments.output or f"{arguments.ethnicity}.db"
    classifications_db = os.path.join(arguments.work_dir, CLASSIFICATIONS_DB)

    return (
        arguments.ethnicity,
        arguments.path,
        arguments.work_dir,
        output,
        classifications_db,
        arguments.rebuild,
    )


def main():
    """
    Takes arguments <ethnicity> <path/to/compressed/files>
    [--work-dir DIR] [--output DB] [--rebuild]
    Classifier used is n2e
    Names and their predicted ethnicity are stored in classifications.db
    together with the compressed files they were read from
    --work-dir holds classifications.db and the filtered files
    --output is the database to populate
    --rebuild drops classifications.db so every name is classified again
    Attach and populate the database using a3k --attach-databases
    Script produces populated tables which can be used for other processes
    """
    ethnicity, path, work_dir, output, classifications_db, rebuild = (
        get_flags()
    )

    if rebuild and os.path.exists(classifications_db):
        os.remove(classifications_db)

    database = create_tables(work_dir)
    names, read_files = extract_names(path, database)
    unclassified_names = unclassified(database, names)

    for chunk in classify_names(unclassified_names):
        with database:
            for row in chunk:
                database.execute(
                    "INSERT OR IGNORE INTO classified_names VALUES (?, ?, ?, ?)",
                    row,
                )

    mark_processed(database, read_files)

    ethnicity_names = filter_names(database, ethnicity)
    print(f"{len(ethnicity_names)} {ethnicity} names")

    output_p = split_files(ethnicity, ethnicity_names, path, work_dir)
    print("")
    print("populating names using a3k")
    populate_database(ethnicity, output_p, classifications_db, output)


if __name__ == "__main__":
    main()
