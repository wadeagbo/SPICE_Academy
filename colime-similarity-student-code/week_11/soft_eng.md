### Software Engineering

The goal of Software Engineering is to build and maintain high-quality, reliable, and efficient computer systems.

**Coding is only a SMALL PART of the Software Engineering Process!** There are lots of engineering principles involved to acheive the goals of SE.

Apart from coding, what are other things that software developers (e.g. DS / python) do?

- Prototyping / Architecture
  - Pseudocode 
  - Class Diagrams
  - See: https://www.visual-paradigm.com/guide/uml-unified-modeling-language/what-is-uml/
- Project Mgmt / Planning / Organisation
  - Agile / Kanban, Scrum
    - Project Mgmt frameworks for managing teams of developers. Breaking up a large problem into smaller ones.
  - User Stories / Project Requirements
  - Flowcharts
- Documentation
  - README file on Github!!!
    - how to run, how to install
      - the presence of `requirements.txt` 
      - running locally -> suggest the user creates a virtual environment (conda)
      - Dockerfile 
  - Docstrings (in functions, in modules, in classes)
  - Sphinx (or MkDocs) -> libraries for creating sophisticated documentation.
    - generated HTML static websites for you automatically. 
- Testing
  - Acceptance Testing
  - Unit Testing -> `pytest`
    - you write code to test your code.
- Maintenance
  - Updates / deprecated APIs / deprecated methods
  - Scraping code invalid 
  - Server Maintenance 
  - Code refactoring 
    - constantly finding ways to make your code more dynamic, more modular, more understandable.
- Infrastructure
  - deciding on technology stack
  - how much volume / load / computing power you will require, e.g. when deciding on which cloud services to use.

---



Software Metaphors:

---

**Cathedral**: Top-down approach to writing code / building. Massive teams.

**Bazaar**: Looking around on stack overflow, taking bits and pieces from other sources, fitting them together as we go. Open-source / ground-up. The majority of code that we use. See **The Cathedral and the Bazaar** -> really good book recommendation. 

**Garden**: LOTS of maintenance. Software has a life of its own. Software becomes more useless / deteriorates over time if not constantly maintained. Lehmann's Laws of Software. Trimming branches / mowing the lawn / deleting lines of code is sometimes BETTER than writing code / adding more stuff.

What are the most important SE tools you need in Python Data Science, besides the DS "tech stack" (e.g. pandas, sklearn, TF) itself?

- Subject Matter Knowledge / Academic Background

  - Abstraction. People who think they're bad at math are not actually bad at math. they MIGHT be bad at abstraction. 
  - Statistics / data structures.

- Version Control

  - `git` 

- Bash / Command line

- **Automated Testing**

  - `pytest`

- IDE / Editor

  - Microsoft Word :)
  - Interactive debugger / variable explorer / linting / auto-formatting 

- Databases!

  - `SQL` -> pgexercises.com !!! 

- **Code Packaging** -> `setuptools` / `PyPI` 

  - how do I turn my code into a pip-installable package so that it can be installed on other machines?

- Cloud Computing

  - AWS / GCP / Azure 

- Docker 

- **Continuous Integration** -> `GitHub Actions`

- **Project Templates**: ` cookiecutter`

- **Profiling Code** (i.e. testing the efficiency / speed of your code) `cProfile`

- Communication / Presenting Results

  





