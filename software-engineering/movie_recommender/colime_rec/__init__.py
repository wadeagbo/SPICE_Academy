"""A tiny movie recommender, written to practise turning code into an installable package."""

from .classes import RecommenderClass
from .subfolder.data import MOVIES

__all__ = ["MOVIES", "RecommenderClass"]
