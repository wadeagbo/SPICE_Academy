import random


class RecommenderClass:
    """Groups the movie recommender functions."""

    def __init__(self, movie_list):
        self.movies = list(movie_list)

    def recommend_movie(self):
        """Recommend a random movie from the list."""
        return random.choice(self.movies)
