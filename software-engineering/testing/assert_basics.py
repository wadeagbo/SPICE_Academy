"""How assert works: it does nothing when the condition is true and raises AssertionError otherwise.

Run with:  python assert_basics.py
"""

assert 10 == 10, "This line never fails"
assert len("any string") == len("other text"), "Both strings have 10 characters"

try:
    assert 1 + 1 == 3, "1 + 1 is not 3"
except AssertionError as err:
    print(f"AssertionError raised as expected: {err}")
