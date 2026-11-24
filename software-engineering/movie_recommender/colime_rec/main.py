"""Command-line entry point. Run with:  python -m colime_rec.main"""

from colime_rec import MOVIES, RecommenderClass


def main():
    print(RecommenderClass(MOVIES).recommend_movie())


if __name__ == "__main__":
    main()
