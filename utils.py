from typing import TypeVar
from gloe import transformer


T = TypeVar("T")
V = TypeVar("V")


@transformer
def pick_first(input: tuple[T, V]) -> T:
    return input[0]


@transformer
def pick_second(input: tuple[T, V]) -> V:
    return input[1]


@transformer
def print_data(input):
    return input


@transformer
def zip_to_one(input: tuple[list[T], V]) -> list[tuple[T, V]]:
    """
    Performs a zip operation of all the elements on the left side of a tuple to the
    same right side element
    """
    return [(element, input[1]) for element in input[0]]


def is_valid_word(s: str) -> bool:
    return all(part.isalnum() and not part.isnumeric() for part in s.split("-"))
