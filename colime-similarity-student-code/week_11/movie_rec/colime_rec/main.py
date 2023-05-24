from subfolder.data import MOVIES    #from sklearn.linear_models import LinearRegression
from classes import RecommenderClass


rec = RecommenderClass(MOVIES)
print(rec.recommend_movie())