"""Links work_authors with their predicted ethnicity"""

from itertools import islice
from collections.abc import Iterable

import apsw
from n2e import predict_ethnicities

from alexandria3k.common import ensure_table_exists, log_sql, set_fast_writing
from alexandria3k import perf
from alexandria3k.db_schema import ColumnMeta, TableMeta

table = [
    TableMeta(
        "author_ethnicities",
        columns=[
            ColumnMeta("work_id"),
            ColumnMeta("work_author_id"),
            ColumnMeta("ethnicity"),
            ColumnMeta("confidence"),
        ],
    ),
]


MODEL = "28_nationalities_english_once"
CHUNK_SIZE = 100000
BATCH_SIZE = 1024


def classify_names(names: Iterable):
    """
    Handles classifying names using the n2e classfier.
    Yields one chunk at a time to prevent memory issues

    :param names: the list of names that will get classfied
    :type names: list

    :yield: yields a chunk of (given, family, ethnicity, confidence)

    """

    perf.log("running classifier")

    while chunk_names := list(islice(names, CHUNK_SIZE)):

        predictions = predict_ethnicities(
            [f"{given} {family}" for given, family in chunk_names],
            batch_size=BATCH_SIZE,
            model=MODEL,
        )
        chunk = [
            (given, family, ethnicity, confidence)
            for (given, family), (ethnicity, confidence) in zip(
                chunk_names, predictions
            )
        ]
        perf.log(f"classified {len(chunk)} names.")
        yield chunk


def create_author_ethnicities_table(database_path):
    """Creates and links author_ethnicities table to the database"""
    database = apsw.Connection(database_path)

    ensure_table_exists(database, "work_authors")
    database.execute(log_sql("DROP TABLE IF EXISTS author_ethnicities"))
    database.execute(log_sql(table[0].table_schema()))
    set_fast_writing(database)

    database.execute(
        "CREATE INDEX IF NOT EXISTS idx_work_authors_name "
        "ON work_authors(given, family)"
    )

    # Returns the number of entries, for logs
    total_names = list(database.execute("SELECT COUNT(*) FROM work_authors"))[
        0
    ][0]

    query = """
            SELECT DISTINCT given, family FROM work_authors 
            WHERE given IS NOT NULL AND family IS NOT NULL
            """
    names = database.execute(query)

    for classifications in classify_names(names):
        perf.log(f"{len(classifications)}/{total_names} names.")
        with database:
            database.executemany(
                """
                INSERT INTO author_ethnicities
                SELECT work_id, work_authors.id, ?, ?
                FROM work_authors
                WHERE given IS ? AND family IS ?
                """,
                [
                    (ethnicity, confidence, given, family)
                    for given, family, ethnicity, confidence in classifications
                ],
            )


def process(database_path):
    """
    Process creates and links the author_ethinicities_table with the populated dataset
    Table consists of (work_id, work_author_id, ethnicity, confidence_score)
    """

    create_author_ethnicities_table(database_path)
    perf.log("Created author_ethnicities table")
