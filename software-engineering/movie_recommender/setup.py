from setuptools import find_packages, setup

setup(
    name="colime_rec",
    version="0.1.0",
    description="A tiny movie recommender, written to practise Python packaging",
    packages=find_packages(),
    python_requires=">=3.9",
    entry_points={"console_scripts": ["colime-rec=colime_rec.main:main"]},
)
