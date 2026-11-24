# colime_rec

A tiny movie recommender, written to practise turning Python code into a package that can be installed
with pip. ("Colime similarity" was the name of my SPICED cohort.)

```bash
pip install -e .
colime-rec                   # prints a random movie
python -m colime_rec.main    # same thing
```

```python
from colime_rec import MOVIES, RecommenderClass
RecommenderClass(MOVIES).recommend_movie()
```
