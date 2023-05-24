import random

class RecommenderClass():
    """Class for grouping all our movie recommender-functions"""
    
    def __init__(self, movie_list):
        self.movies = movie_list
        self.attr = None 
    
   
    def recommend_movie(self):
        """Randomly recommend movies from a given list"""
        result = random.choice(self.movies)
        return result 
    
   
    def nmf(self, n):
        """coming soon in version 2.0"""
        self.attr = n


if __name__ == '__main__':
    c = RecommenderClass([1, 2, 3])
    print(c.recommend_movie())