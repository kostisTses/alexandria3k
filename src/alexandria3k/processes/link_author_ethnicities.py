"""Links work_authors with their predicted ethnicity"""
import apsw

from alexandria3k.common import ensure_table_exists, log_sql, set_fast_writing

from alexandria3k.db_schema import ColumnMeta, TableMeta
from alexandria3k.author_name_disambiguation.ethnicity_utils import classify_names

table = [
    TableMeta(
        "author_ethnicities",
        columns=[
            ColumnMeta("work_id"),
            ColumnMeta("work_author_id"),
            ColumnMeta("ethnicity"),
            ColumnMeta("confidence")
        ],
    ),
]


def create_author_ethnicities_table(database_path):
    """Creates and links author_ethnicities table to the database"""
    database = apsw.Connection(database_path)

    ensure_table_exists(database, "work_authors")
    ensure_table_exists(database, "works")
    database.execute(log_sql("DROP TABLE IF EXISTS author_ethnicities"))
    database.execute(log_sql(table[0].table_schema()))
    set_fast_writing(database)

    database.execute(
        "CREATE INDEX IF NOT EXISTS idx_work_authors_name "
        "ON work_authors(given, family)"
    )

    names = list(database.execute(
        """
        SELECT DISTINCT given, family FROM work_authors 
        WHERE given IS NOT NULL AND family IS NOT NULL
        """
    ))


    for classifications in classify_names(names):
        with database: 
            for given, family, ethnicity, confidence in classifications:
                database.execute(
                    """
                    INSERT INTO author_ethnicities
                    SELECT work_id, id, ?, ?
                    FROM work_authors
                    WHERE given IS ? AND family IS ?
                    """,
                    (ethnicity, confidence, given, family),
                )



def process(database_path):
    """
    Process creates and links the author_blocks_table with the populated dataset
    Table consists of (work_author_id, normalised_name, normalised_family_name, work_id, block_key)
    For each author in the database, his name will be passed through a name-normalization function,
    based on that output each author will be put into a block with that id,
    reducing comparisons to just each authors in the same block to distinguish.
    block_key is consisted of the normalised_family_name + first inital of normalised name
    """

    create_author_ethnicities_table(database_path)
    print("Created author_ethnicities table")