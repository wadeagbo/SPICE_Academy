"""Unit tests for funcs.py. Run with:  pytest"""

import pytest

from funcs import add_one, recommend_movie

EXAMPLES = [(5, 6), (100, 101), (5.3, 6.3), (-100, -99)]
MOVIES = ["Harry Potter and the Colime Similarity", "Star Wars: The colimes awaken", "Lord of the colimes"]


@pytest.mark.parametrize(("inp", "expected"), EXAMPLES)
def test_add_one(inp, expected):
    assert add_one(inp) == pytest.approx(expected)


def test_recommendation_is_an_upper_case_title_from_the_list():
    result = recommend_movie(MOVIES)
    assert isinstance(result, str)
    assert result.isupper()
    assert result in [m.upper() for m in MOVIES]


def test_wrong_input_type():
    with pytest.raises(TypeError):
        recommend_movie("Hello")


def test_empty_list():
    with pytest.raises(ValueError):
        recommend_movie([])
