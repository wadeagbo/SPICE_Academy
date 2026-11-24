"""Small functions used to practise unit testing with pytest."""

import random


def add_one(x):
    """Return x + 1."""
    return x + 1


def recommend_movie(movie_list):
    """Pick a random movie from a list and return its title in upper case.

    Raises TypeError if movie_list is not a list, and ValueError if it is empty.
    """
    if not isinstance(movie_list, list):
        raise TypeError("movie_list has to be a list of titles")
    if not movie_list:
        raise ValueError("movie_list is empty")
    return random.choice(movie_list).upper()
