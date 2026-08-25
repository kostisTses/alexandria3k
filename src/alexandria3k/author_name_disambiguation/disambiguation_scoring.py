from datasketch import MinHash
from rapidfuzz.distance import JaroWinkler

from alexandria3k.author_name_disambiguation.disambiguation_util import (
    Author,
    get_ngrams,
    jaccard_similarity,
)


class Block_Attr():
    """Every attribute of a block used to compare authors together"""
    def __init__(self, block_key, authors):
        self.block_key = block_key
        self.authors = authors
        self.co_authors = {}
        self.affiliations = {}
        self.publication_years = {}
        self.venues = {}
        self.min_hashes = {}

    def load(self, database):
        self.co_authors = get_co_authors(self.block_key, database)
        self.affiliations = get_affiliations_per_block(self.block_key, database)
        self.publication_years = get_publication_years_per_block(self.block_key, database)
        self.venues = get_venue_per_block(self.block_key, database)
        self.min_hashes = get_min_hashes(self.co_authors)

def get_min_hashes(co_authors_map: dict[int, set[str]], max_threshold=500, min_authors=10):

    min_hash_map = {}

    if not co_authors_map:
        return {}

    if len(co_authors_map) < min_authors:
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


def get_co_authors(key, database):
    """
    Gets all the co-authors of a specific work of an author and stores in a set
    Takes everything from the author_names_blocks table
    """

    co_authors_cursor = database.cursor()

    # Key: work_author_id , Value: set of block_keys corresponding to their co_authors
    coauthor_map: dict[int, set[str]] = {}
    for work_author_id, coauthor_block_key in co_authors_cursor.execute(
        """
        SELECT a.work_author_id, b.block_key
        FROM author_name_blocks a
        JOIN author_name_blocks b ON b.work_id = a.work_id
        WHERE a.block_key = ?
        AND b.work_author_id != a.work_author_id
    """,
        (key,),
    ):
        if work_author_id not in coauthor_map:
            coauthor_map[work_author_id] = set()
        coauthor_map[work_author_id].add(coauthor_block_key)

    return coauthor_map


def get_affiliations_per_block(block_key, database):
    """
    Returns a map consisting of all the affiliations of an author_id in a Block,
    affiliations could be more than one so set is needed,
    returns the n-gram of each affiliations for better comparisons
    get_ngrams implemented in author_name_disambiguation_utils.py
    params:
        block_key, the block key that we are taking authors from
        database,  apsw connection of the populated database

    """
    affiliations_cursor = database.cursor()

    # Key: author_id , Value: set of affiliations
    affiliations_map: dict[int, set[str]] = {}

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
    Queries the database for publication year of the work of each author in a block
    """

    cursor = database.cursor()

    # Key: author_id, Value: publication_year
    publication_year_map: dict[int, int] = {}

    for author_id, published_year in cursor.execute(
        """
    SELECT work_author_id ,published_year  FROM works
    JOIN author_name_blocks ON author_name_blocks.work_id = works.id
    WHERE author_name_blocks.block_key = ? """,
        (block_key,),
    ):
        publication_year_map[author_id] = published_year

    return publication_year_map


def get_venue_per_block(block_key, database):
    """
    Queries the database for venue of the work of each author in block
    """
    cursor = database.cursor()

    # Key: author_id, Value: set of venues
    venue_map: dict[int, set[str]] = {}

    for work_author_id, container_title, shortened_container_title in cursor.execute(
        """
        SELECT work_author_id, works.container_title, works.short_container_title
        FROM works
        JOIN author_name_blocks ON author_name_blocks.work_id = works.id
        WHERE author_name_blocks.block_key = ?
        """,
        (block_key,),
    ):
        venue = container_title or shortened_container_title
        if not venue:
            continue

        if work_author_id not in venue_map:
            venue_map[work_author_id] = set()
        venue_map[work_author_id].update(get_ngrams(venue))
    return venue_map


def score_affiliations(auth1, auth2, block):
    """
    1-3 word n-gram Jaccard similarity of two authors normalized affiliation strings
    auth = set(author_id, author_name, author_work_id )
    """

    author_1_affiliations = block.affiliations.get(auth1.id, set())
    author_2_affiliations = block.affiliations.get(auth2.id, set())

    if not author_1_affiliations or not author_2_affiliations:
        return None

    return jaccard_similarity(author_1_affiliations, author_2_affiliations)


def score_venue(auth1, auth2, block):
    """
    Word n-gram Jaccard similarity of the venues two authors published in.
    """

    author_1_venue = block.venues.get(auth1.id, set())
    author_2_venue = block.venues.get(auth2.id, set())

    if not author_1_venue or not author_2_venue:
        return None

    return jaccard_similarity(author_1_venue, author_2_venue)


def score_coauthors(auth1, auth2, block, weight=1.5):
    """Jaccard similarity of two authors' co-author block-key sets"""

    auth1_coauthors = block.co_authors.get(auth1.id, set())
    auth2_coauthors = block.co_authors.get(auth2.id, set())

    if not auth1_coauthors or not auth2_coauthors:
        return None

    similarity = score_min_hash(auth1, auth2, block)
    if similarity is None:
        similarity = jaccard_similarity(auth1_coauthors, auth2_coauthors)

    return min(1.0, weight * similarity)


def score_name_similarity(auth1, auth2, threshold=0.75):
    """
    Score the name similarity of 2 authors normalised_name
    If the name is the same return 1
    If name1 != name2 find jaroWinkler similarity
    if jarowinkler < threshold return 0
    """

    if auth1.name == auth2.name:
        return 1.0
    name_similarity = JaroWinkler.similarity(auth1.name, auth2.name)

    return 0 if name_similarity < threshold else name_similarity

def score_min_hash(auth1, auth2, block):
    """
    Estimated Jaccard of two authors' co-author sets from their MinHash
    Returns None when the block was not big enough to build them
    """

    min_hash1 = block.min_hashes.get(auth1.id)
    min_hash2 = block.min_hashes.get(auth2.id)

    if min_hash1 is None or min_hash2 is None:
        return None

    return min_hash1.jaccard(min_hash2)

def check_if_co_authors(auth1, auth2):
    """
    Checks if 2 authors are co_authors by comparing if work_id is the same
    """
    return auth1.work_id == auth2.work_id


def check_communities(auth1, auth2):
    "Checks if 2 authors are in the same community based on journals"
    if auth1.community_id is None or auth2.community_id is None:
        return False
    return auth1.community_id != auth2.community_id

def check_year_gap(auth1, auth2, block, max_gap=40):
    """
    Get year gaps where authors made publications
    If there is a big gap between them they are probably not the same person
    """

    year1 = block.publication_years.get(auth1.id)
    year2 = block.publication_years.get(auth2.id)

    if year1 is None or year2 is None:
        return None

    gap = abs(year1 - year2)
    return gap > max_gap

def compare_authors(
    auth1: Author,
    auth2: Author,
    block: Block_Attr,
    threshold=0.67
):
    """This will serve as the scoring function to determine if 2 authors are the same person.
    The scoring function will be calculated based on a couple of criteria:
    - Jaccard similarity on co-author sets of each author (how many co-authors they have in common)
    - Jaro Winkler score , comparing the normalized names
    - Affiliation/venue overlap
    - Year gap
    - Topic overlap using Leiden clustering
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
    jaccard_venue = score_venue(auth1, auth2, block)

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