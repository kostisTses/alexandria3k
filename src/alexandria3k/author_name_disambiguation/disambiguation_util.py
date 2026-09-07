"""
Includes all external utilities that are used in the author_name_disambiguation pipeline
"""

import unicodedata

from sklearn.feature_extraction.text import CountVectorizer


# Adapted from uf-toolkit (https://github.com/hugginsc10/uf-toolkit)
# Copyright (c) Chas Huggins
# Licensed under the MIT License
class UnionFind:
    """UnionFind datastructure implementation taken from github used for merging authors together"""

    def __init__(self, size):
        "init"
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i):
        "finds nodes parent"
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, a, b):
        "unions two nodes"
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            # Union by rank
            if self.rank[root_a] < self.rank[root_b]:
                self.parent[root_a] = root_b
            elif self.rank[root_a] > self.rank[root_b]:
                self.parent[root_b] = root_a
            else:
                self.parent[root_b] = root_a
                self.rank[root_a] += 1

    def connected(self, a, b):
        "checks if two nodes are connected"
        return self.find(a) == self.find(b)

    def count_sets(self):
        "counts number of sets"
        return sum(1 for i in range(len(self.parent)) if i == self.parent[i])

    def get_set_elements(self, i):
        "gets all elements of a set"
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
    """Custom normalization function used for normalizing author_names"""
    tmp = "".join(
        c
        for c in unicodedata.normalize("NFKD", s)
        if unicodedata.category(c) != "Mn" and c.isalpha()
    )
    return tmp.lower().strip()


ngram = CountVectorizer(ngram_range=(1, 1)).build_analyzer()


def get_ngrams(text):
    "gets n_grams of a sentence"
    return set(ngram(text))
