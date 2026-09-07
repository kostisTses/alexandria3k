"""
Includes all of the classes/methods relative to comparing and scoring two authors,
based on their similarities

- AuthorAttr class contains information of an author
- BlockAttr  class contains information about all authors in a block
- get_*       methods gets all the attributes that are used to compare authors,
              runs once for all the authors in one block

- score_*     methods score the similarity of the attributes,
              if avg of all scores exceed a threashold we say they are the same

- check_*     methods are boolean and check for cheap conditions that rules out a match

- compare_authors method is the main caller of all the scoring methods
"""

from typing import NamedTuple

from datasketch import MinHash
from rapidfuzz.distance import JaroWinkler

from alexandria3k.author_name_disambiguation.disambiguation_util import (
    get_ngrams,
    jaccard_similarity,
)


class AuthorAttr(NamedTuple):
    """One author mention within a block"""

    work_author_id: int
    name: str
    work_id: int
    community_id: int


class BlockAttr:
    """Every attribute of a block used to compare authors together"""

    # pylint: disable=too-few-public-methods

    def __init__(self, block_key, authors):
        self.block_key = block_key
        self.authors = authors
        self.co_authors = {}
        self.affiliations = {}
        self.publication_years = {}
        self.journals = {}
        self.min_hashes = {}

    def load(self, database):
        "loads all attributes of each block"

        self.co_authors = get_co_authors(self.block_key, database)
        self.affiliations = get_affiliations_per_block(
            self.block_key, database
        )
        self.publication_years = get_publication_years_per_block(
            self.block_key, database
        )
        self.journals = get_journal_per_block(self.block_key, database)
        self.min_hashes = get_min_hashes(self.co_authors)


def get_min_hashes(co_authors_map: dict[int, set], max_threshold=500):
    """
    Calculates min_hash for authors with large co_authors count

    :param co_authors_map: map of co_authors
    :type co_authors_map:  dict[int, set]

    :param max_threshold:  determines threashold of large co_author count, defaults to 500

    :return:               dict of  key: author_id, value: min_hash
    """

    min_hash_map = {}

    if not co_authors_map:
        return {}

    max_coauth = max(len(co_authors) for co_authors in co_authors_map.values())
    if max_coauth <= max_threshold:
        return {}

    min_hash_map = {}
    for author_id, co_authors in co_authors_map.items():
        if len(co_authors) > max_threshold:
            min_hash = MinHash(num_perm=128)
            min_hash.update_batch([c.encode("utf-8") for c in co_authors])
            min_hash_map[author_id] = min_hash

    return min_hash_map


def get_co_authors(block_key, database):
    """
    Finds all co_authors of each author in a block
    Stores everything in a dict

    :param block_key: the block_key to be checked
    :type block_key:  str

    :param database:  the database we are working in
    :type database:   Connection

    :return: dict of key: work_author_id, value: set of each co_authors block_key
    """

    co_authors_cursor = database.cursor()

    # Key: author_id , Value: set of co_author block_key
    coauthor_map = {}

    for work_author_id, coauthor_block_key in co_authors_cursor.execute(
        """
        SELECT a.work_author_id, b.block_key
        FROM author_name_blocks a
        JOIN author_name_blocks b ON b.work_id = a.work_id
        WHERE a.block_key = ?
        AND b.work_author_id != a.work_author_id
    """,
        (block_key,),
    ):
        if work_author_id not in coauthor_map:
            coauthor_map[work_author_id] = set()
        coauthor_map[work_author_id].add(coauthor_block_key)

    return coauthor_map


def get_affiliations_per_block(block_key, database):
    """
    Finds all affiliations of each author in a block
    Stores everything in a dict
    Using n_grams for more precise comparisons

    :param block_key: the block_key to be checked
    :type block_key:  str

    :param database:  the database we are working in
    :type database:   Connection

    :return: dict of key: work_author_id, value: set of affiliations
    """
    affiliations_cursor = database.cursor()

    # Key: author_id , Value: set of affiliations
    affiliations_map = {}

    for author_id, affiliation_name in affiliations_cursor.execute(
        """SELECT author_id, name FROM author_affiliations
        JOIN author_name_blocks ON  author_id = author_name_blocks.work_author_id
        WHERE author_name_blocks.block_key = ? """,
        (block_key,),
    ):
        if not affiliation_name:
            continue
        if author_id not in affiliations_map:
            affiliations_map[author_id] = set()
        affiliations_map[author_id].update(get_ngrams(affiliation_name))
    return affiliations_map


def get_publication_years_per_block(block_key, database):
    """
    Finds all publication years of each author in a block
    Stores everything in a dict

    :param block_key: the block_key to be checked
    :type block_key:  str

    :param database:  the database we are working in
    :type database:   Connection

    :return: dict of key: work_author_id, value: publication_year
    """

    cursor = database.cursor()

    # Key: author_id, Value: publication_year
    publication_year_map = {}

    for author_id, published_year in cursor.execute(
        """
    SELECT work_author_id ,published_year  FROM works
    JOIN author_name_blocks ON author_name_blocks.work_id = works.id
    WHERE author_name_blocks.block_key = ? """,
        (block_key,),
    ):
        publication_year_map[author_id] = published_year

    return publication_year_map


def get_journal_per_block(block_key, database):
    """
    Finds all journals of each author in a block
    Stores everything in a dict

    :param block_key: the block_key to be checked
    :type block_key:  str

    :param database:  the database we are working in
    :type database:   Connection

    :return: dict of key: work_author_id, value: set of journals
    """

    cursor = database.cursor()

    # Key: author_id, Value: set of journals
    journal_map = {}

    for (
        work_author_id,
        container_title,
        shortened_container_title,
    ) in cursor.execute(
        """
        SELECT work_author_id, works.container_title, works.short_container_title
        FROM works
        JOIN author_name_blocks ON author_name_blocks.work_id = works.id
        WHERE author_name_blocks.block_key = ?
        """,
        (block_key,),
    ):
        journal = container_title or shortened_container_title
        if not journal:
            continue

        if work_author_id not in journal_map:
            journal_map[work_author_id] = set()
        journal_map[work_author_id].update(get_ngrams(journal))
    return journal_map


def score_affiliations(auth1: AuthorAttr, auth2: AuthorAttr, block: BlockAttr):
    """
    Scores n-gram jaccard similarity of two authors affiliation sets

    :param auth1: Author1
    :type auth1:  AuthorAttr
    :param auth2: Author2
    :type auth2:  AuthorAttr
    :param block: Block
    :type block:  BlockAttr
    :return:      Jaccard Similarity score
    """

    author_1_affiliations = block.affiliations.get(auth1.work_author_id, set())
    author_2_affiliations = block.affiliations.get(auth2.work_author_id, set())

    if not author_1_affiliations or not author_2_affiliations:
        return None

    return jaccard_similarity(author_1_affiliations, author_2_affiliations)


def score_journals(auth1: AuthorAttr, auth2: AuthorAttr, block: BlockAttr):
    """
    Scores n-gram jaccard similarity of two authors journals sets

    :param auth1: Author1
    :type auth1:  AuthorAttr
    :param auth2: Author2
    :type auth2:  AuthorAttr
    :param block: Block
    :type block:  BlockAttr
    :return:      Jaccard Similarity score
    """

    author_1_journals = block.journals.get(auth1.work_author_id, set())
    author_2_journals = block.journals.get(auth2.work_author_id, set())

    if not author_1_journals or not author_2_journals:
        return None

    return jaccard_similarity(author_1_journals, author_2_journals)


def score_coauthors(
    auth1: AuthorAttr, auth2: AuthorAttr, block: BlockAttr, weight=1.5
):
    """
    Scores jaccard similarity of two authors co_authors sets
    If co-authors count is too large, calculate min-hash instead
    This comparison metric is weighted

    :param auth1: Author1
    :type auth1:  AuthorAttr
    :param auth2: Author2
    :type auth2:  AuthorAttr
    :param block: Block
    :type block:  BlockAttr
    :return:      Jaccard Similarity score
    """

    auth1_coauthors = block.co_authors.get(auth1.work_author_id, set())
    auth2_coauthors = block.co_authors.get(auth2.work_author_id, set())

    if not auth1_coauthors or not auth2_coauthors:
        return None

    similarity = score_min_hash(auth1, auth2, block)
    if similarity is None:
        similarity = jaccard_similarity(auth1_coauthors, auth2_coauthors)

    return min(1.0, weight * similarity)


def score_name_similarity(
    auth1: AuthorAttr, auth2: AuthorAttr, threshold=0.75
):
    """ "
    Scores JaroWinkler name similarity of two author names

    :return: Jarowinkler score
    """

    if auth1.name == auth2.name:
        return 1.0
    name_similarity = JaroWinkler.similarity(auth1.name, auth2.name)

    return 0 if name_similarity < threshold else name_similarity


def score_min_hash(auth1: AuthorAttr, auth2: AuthorAttr, block: BlockAttr):
    """
    Scores MinHash of two authors if their co_author count is large enough

    :param auth1: Author1
    :type auth1:  AuthorAttr
    :param auth2: Author2
    :type auth2:  AuthorAttr
    :param block: Block
    :type block:  BlockAttr
    :return:      MinHash score
    """

    min_hash1 = block.min_hashes.get(auth1.work_author_id)
    min_hash2 = block.min_hashes.get(auth2.work_author_id)

    if min_hash1 is None or min_hash2 is None:
        return None

    return min_hash1.jaccard(min_hash2)


def check_if_co_authors(auth1: AuthorAttr, auth2: AuthorAttr) -> bool:
    """
    Checks if 2 authors are co_authors by comparing if work_id is the same
    """
    return auth1.work_id == auth2.work_id


def check_communities(auth1: AuthorAttr, auth2: AuthorAttr) -> bool:
    "Checks if 2 authors are in the same community of journals"

    if auth1.community_id is None or auth2.community_id is None:
        return False
    return auth1.community_id != auth2.community_id


def check_year_gap(
    auth1: AuthorAttr, auth2: AuthorAttr, block: BlockAttr, max_gap=40
) -> bool:
    """
    Checks if two authors year_gap
    If its large enough consider them different
    """

    year1 = block.publication_years.get(auth1.work_author_id)
    year2 = block.publication_years.get(auth2.work_author_id)

    if year1 is None or year2 is None:
        return None

    gap = abs(year1 - year2)
    return gap > max_gap


def compare_authors(
    auth1: AuthorAttr, auth2: AuthorAttr, block: BlockAttr, threshold=0.67
):
    """
    Scores a pair of two authors similarity based on:
    - Jaccard similarity on co-author sets
    - Jaro Winkler normalized name score
    - Affiliation overlap
    - Journal overlap
    - Year gap
    - Journal community/topic overlap using Leiden clustering

    :param auth1: Auhtor 1
    :type auth1:  AuthorAttr

    :param auth2: Author 2
    :type auth2:  AuthorAttr

    :param block: Block
    :type block:  BlockAttr

    :param threshold: Minimum merging threashold, defaults to 0.67

    :return: Similarity score
    """

    if check_communities(auth1, auth2):
        return 0
    if check_if_co_authors(auth1, auth2):
        return 0
    if check_year_gap(auth1, auth2, block):
        return 0

    name_similarity = score_name_similarity(auth1, auth2)
    jaccard_affiliations = score_affiliations(auth1, auth2, block)
    jaccard_coauthors = score_coauthors(auth1, auth2, block)
    jaccard_venue = score_journals(auth1, auth2, block)

    # calculate confidence score
    scores = [
        name_similarity,
        jaccard_affiliations,
        jaccard_coauthors,
        jaccard_venue,
    ]
    valid_scores = [s for s in scores if s is not None]
    avg = sum(valid_scores) / len(valid_scores)

    return avg if avg >= threshold else 0
