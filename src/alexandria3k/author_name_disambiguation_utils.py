from itertools import groupby
from typing import NamedTuple
import unicodedata

from rapidfuzz.distance import JaroWinkler
from datasketch import MinHash
from sklearn.feature_extraction.text import CountVectorizer

class Author(NamedTuple):
    """One author mention within a block"""
    id: int
    name: str
    work_id: int
    community_id: int

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

def get_min_hashes(co_authors_map: dict[int, set[str]], max_threashold=500, min_authors=10):

    min_hash_map = {}

    if not co_authors_map:
        return {}

    if len(co_authors_map) < min_authors:
        return {}

    max_coauth = max(len(co_authors) for co_authors in co_authors_map.values())
    if max_coauth <= max_threashold:
        return {}

    min_hash_map = {}
    for author_id, co_authors in co_authors_map.items():
        if len(co_authors) > max_threashold:
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


# Adapted from uf-toolkit (https://github.com/hugginsc10/uf-toolkit)
# Copyright (c) Chas Huggins
# Licensed under the MIT License
class UnionFind:
    """UnionFind datastructure implementation taken from github used for merging authors together"""

    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [0] * size

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, a, b):
        rootA = self.find(a)
        rootB = self.find(b)

        if rootA != rootB:
            # Union by rank
            if self.rank[rootA] < self.rank[rootB]:
                self.parent[rootA] = rootB
            elif self.rank[rootA] > self.rank[rootB]:
                self.parent[rootB] = rootA
            else:
                self.parent[rootB] = rootA
                self.rank[rootA] += 1
    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def count_sets(self):
        return sum(1 for i in range(len(self.parent)) if i == self.parent[i])

    def get_set_elements(self, i):
        root = self.find(i)
        return [x for x in range(len(self.parent)) if self.find(x) == root]


def jaccard_similarity(set_a, set_b):
    """Custom jaccard similarity used for comparing authors"""
    if set_a and set_b:
        union = set_a | set_b
        jaccard = len(set_a & set_b) / len(union)
    else:
        jaccard = 0

    return jaccard

def normalized(s: str):
    """Custom normalization function used for normalizing author_names """
    tmp = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if unicodedata.category(c) != "Mn" and c.isalpha() 
    )
    return tmp.lower().strip()


ngram = CountVectorizer(ngram_range=(1, 1)).build_analyzer()

def get_ngrams(text):
    return set(ngram(text))