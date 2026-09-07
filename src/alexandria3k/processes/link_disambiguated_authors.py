"""Links authors with their predicted disambiguated identity"""

from itertools import groupby
from multiprocessing import Pool
import time
from os import cpu_count

import apsw

from alexandria3k.common import ensure_table_exists, log_sql, set_fast_writing

from alexandria3k.author_name_disambiguation.disambiguation_util import (
    UnionFind,
)
from alexandria3k.author_name_disambiguation.disambiguation_init import (
    create_author_blocks_table,
)
from alexandria3k.author_name_disambiguation.disambiguation_scoring import (
    AuthorAttr,
    BlockAttr,
    compare_authors,
)

# from alexandria3k import perf
from alexandria3k.db_schema import ColumnMeta, TableMeta

table = [
    TableMeta(
        "disambiguated_authors",
        columns=[
            ColumnMeta("work_author_id"),
            ColumnMeta("disambiguated_author_id"),
            ColumnMeta("confidence_score"),
        ],
    ),
]


def group_by_signature(authors, block: BlockAttr):
    """Groups authors by their (name, co-authors, affiliations, year) signature."""
    groups = {}
    for author in authors:
        signature = (
            author.name,
            frozenset(block.co_authors.get(author.work_author_id, set())),
            frozenset(block.affiliations.get(author.work_author_id, set())),
            block.publication_years.get(author.work_author_id),
        )
        if signature not in groups:
            groups[signature] = []
        groups[signature].append(author)

    return groups


# pylint: disable-next=too-many-locals
def process_block(block_key, grouped_authors, database_path, database=None):
    """
    Main loop for comparing each author with every other in the block
    Returns a list of each author entry in the database
    entry = tuple(work_author_id, disambiguated_author_id, confidence_score)
    for comparison details check compare_authors
    """
    if database is None:
        database = apsw.Connection(database_path)

    disambiguated_authors_list: list[tuple] = []

    authors = [
        AuthorAttr(
            work_author_id=row[1],
            name=row[2],
            work_id=row[3],
            community_id=row[4],
        )
        for row in grouped_authors
    ]

    block = BlockAttr(block_key, authors)
    block.load(database)

    # unionfind datastructure for merging
    # Groups authors by hashed signature, compare only first author of each group
    # score = 1 for authors in the same group
    groups = group_by_signature(authors, block)
    union_find = UnionFind(len(groups))
    scores = [0.0] * len(groups)

    representatives = [group[0] for group in groups.values()]
    for i, group in enumerate(groups.values()):
        if len(group) > 1:
            scores[i] = 1.0

    for i, representative_1 in enumerate(representatives):
        for j, representative_2 in enumerate(representatives):

            if representative_1 == representative_2:
                continue

            if score := compare_authors(
                representative_1, representative_2, block
            ):
                union_find.union(i, j)
                scores[i] = max(scores[i], score)
                scores[j] = max(scores[j], score)

    for i, signature in enumerate(groups.keys()):
        root = union_find.find(i)
        disambiguated_author_id = representatives[root].work_author_id
        confidence_score = scores[i]
        for author in groups[signature]:
            disambiguated_authors_entry = (
                author.work_author_id,
                disambiguated_author_id,
                confidence_score,
            )
            disambiguated_authors_list.append(disambiguated_authors_entry)

    return disambiguated_authors_list


def process_chunk(chunk, database_path):
    "Processes a number of blocks in parallel instead of one block at a time"

    database = apsw.Connection(database_path)
    result = []
    for block_key, grouped_authors, _ in chunk:
        result.append(
            process_block(block_key, grouped_authors, database_path, database)
        )
    return result


def ethnicity_query(ethnicity=None):
    "Query to get specific ethnicity"
    join = ""
    where = ""
    parameters = ()
    if ethnicity:
        join = "JOIN author_ethnicities USING (work_author_id, work_id)"
        where = "WHERE ethnicity = ?"
        parameters = (ethnicity,)

    query = f"""
        SELECT block_key, work_author_id, normalized_name, work_id, community_id
        FROM author_name_blocks
        {join}
        {where}
        ORDER BY block_key
    """
    return query, parameters


def build_block_args(
    block_cursor, database_path, big_block_threshold=50, ethinicity=None
):
    """Builds arguments for each block to be processed"""

    big_block_args = []
    small_block_args = []
    single_block_results = []

    query, params = ethnicity_query(ethinicity)

    for block_key, grouped_authors in groupby(
        block_cursor.execute(query, params), key=lambda row: row[0]
    ):
        grouped_authors = list(grouped_authors)
        if len(grouped_authors) < 2:
            work_author_id = grouped_authors[0][1]
            single_block_results.append(
                [(work_author_id, work_author_id, 1.0)]
            )

        elif len(grouped_authors) > big_block_threshold:
            big_block_args.append((block_key, grouped_authors, database_path))
        else:
            small_block_args.append(
                (block_key, grouped_authors, database_path)
            )

    return big_block_args, small_block_args, single_block_results


def process_blocks_parallel(block_cursor, database_path):
    """
    Function is called when create_disambiguated_authors_table "parallelised" flag = True
    Processes blocks in parallel rather than sequencially
    Bigger blocks over a threshold are processed in parallel
    Smaller blocks are divided into chunks and handled in parallel
    Concurrency occurs with processes not threads
    Returns a list of block entries in form of a list of tuples
    """

    mark = time.perf_counter()
    big_block_args = []
    small_block_args = []
    single_block_results = []

    big_block_args, small_block_args, single_block_results = build_block_args(
        block_cursor, database_path
    )

    print(f"populated args {time.perf_counter() - mark:.2f}s")
    big_block_args.sort(key=lambda args: len(args[1]), reverse=True)
    small_block_args.sort(key=lambda args: len(args[1]), reverse=True)

    mark = time.perf_counter()

    # big blocks get processed one for each process
    with Pool() as pool:
        results_big = pool.starmap(process_block, big_block_args, chunksize=1)

    print(f"big blocks {time.perf_counter() - mark:.2f}s ")

    mark = time.perf_counter()

    # split small blocks into a number of workers and process each of them concurrently
    workers_count = cpu_count()
    chunks = [small_block_args[j::workers_count] for j in range(workers_count)]

    with Pool(processes=workers_count) as pool:
        results_small = pool.starmap(
            process_chunk, [(chunk, database_path) for chunk in chunks]
        )

    results_small = [row for batch in results_small for row in batch]
    print(f"small blocks {time.perf_counter() - mark:.2f}s ")
    results = results_big + results_small + single_block_results
    return results


def process_blocks_sequential(block_cursor, database_path, database):
    """
    Function is called when create_disambiguated_authors_table "parallelised" flag = False
    Processes blocks sequencially rather than in parallel
    Returns a list of block entries in form of a list of tuples
    """
    args = []
    query, params = ethnicity_query()
    for block_key, grouped_authors in groupby(
        block_cursor.execute(query, params), key=lambda row: row[0]
    ):
        grouped_authors = list(grouped_authors)
        args.append((block_key, grouped_authors, database_path))

    results = []
    for block_key, grouped_authors, path in args:
        results.append(
            process_block(block_key, grouped_authors, path, database=database)
        )

    return results


def create_disambiguated_authors_table(database_path, parallelised=True):
    """Creates and links disambiguated_authors table.
    Takes as input the database path and checks if author_name_blocks table exists
    """

    database = apsw.Connection(database_path)
    database.execute(log_sql("DROP TABLE IF EXISTS disambiguated_authors"))
    database.execute(log_sql(table[0].table_schema()))

    ensure_table_exists(database, "author_name_blocks")
    ensure_table_exists(database, "author_affiliations")
    ensure_table_exists(database, "work_authors")
    ensure_table_exists(database, "works")
    # perf.log("disambiguated_authors table created")

    database.execute(
        log_sql("CREATE INDEX IF NOT EXISTS idx_works_id ON works(id)")
    )
    database.execute(
        log_sql(
            "CREATE INDEX IF NOT EXISTS idx_work_authors_id ON work_authors(id)"
        )
    )
    database.execute(log_sql("""
            CREATE INDEX IF NOT EXISTS idx_author_affiliations_author_id
            ON author_affiliations(author_id)
            """))
    # perf.log("created works/work_authors indexes")

    block_cursor = database.cursor()
    insert_cursor = database.cursor()

    # perf.log("starting block comparison loop")

    if parallelised:
        results = process_blocks_parallel(block_cursor, database_path)
    else:
        results = process_blocks_sequential(
            block_cursor, database_path, database
        )

    set_fast_writing(database)
    for rows in results:
        for row in rows:
            insert_cursor.execute(
                "INSERT INTO disambiguated_authors VALUES (?, ?, ?)",
                row,
                prepare_flags=apsw.SQLITE_PREPARE_PERSISTENT,
            )

    # perf.log("finished comparing all blocks")


def process(database_path):
    """Process creates the disambiguated_authors table for the author name disambiguation pipeline.
    Input will be the author_names_block table created in
    /alexandria3k/src/alexandria3k/processes/link_author_blocks.py for less comparisons.
    For every entry in a block , compares every pair of authors through a scoring function
    which they will be compared by some criteria mentioned in compare_authors
    Process will return a table that contains work_author_id ,
    disambiguated_author_id , confidence_score
    where disambiguated_author_id is the id of the disambiguated author"""

    timer = time.perf_counter()

    create_author_blocks_table(database_path)
    print(f"Built author-blocks in {time.perf_counter() - timer:2f}s")

    create_disambiguated_authors_table(database_path)
    print(f"Built author blocks table in {time.perf_counter() - timer:2f}s")
