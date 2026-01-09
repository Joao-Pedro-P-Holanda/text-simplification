from collections.abc import Iterable
from typing import TypeVar
from gloe import transformer
import re


T = TypeVar("T")
V = TypeVar("V")


@transformer
def pick_first(input: tuple[T, V]) -> T:
    return input[0]


@transformer
def pick_second(input: tuple[T, V]) -> V:
    return input[1]


@transformer
def convert_to_list(input: T) -> list[T]:
    return [input]


@transformer
def zip_to_one(input: tuple[list[T], V]) -> list[tuple[T, V]]:
    """
    Performs a zip operation of all the elements on the left side of a tuple to the
    same right side element
    """
    return [(element, input[1]) for element in input[0]]


@transformer
def strip_special_characters(text: str) -> str:
    return re.sub(re.compile(r"\*+|-{2,}|#+"), "", text)


@transformer
def remove_markdown_tables(text: str) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        if not re.match(r"\|.*\|", line.strip()):
            lines.append(line)

    return "".join(lines)


@transformer
def remove_enumerations(text: str) -> str:
    roman_numerals_regex = re.compile(r"^[i,v,x]+(\.|\)| -)", re.IGNORECASE)
    # a) A) and 1) are matched
    alphabetic_regex = re.compile(r"^(\w\))", re.IGNORECASE)
    # 1. 1- 1.2 and 1 - are matched
    numeric_regex = re.compile(r"^(\d\.*)+")

    lines = []

    for line in text.splitlines(keepends=True):
        without_romans = re.sub(roman_numerals_regex, "", line)
        lines.append(without_romans.strip(" "))

    without_romans = "".join(lines)
    lines = []

    for line in without_romans.splitlines(keepends=True):
        without_alphabetic = re.sub(alphabetic_regex, "", line)
        lines.append(without_alphabetic.strip(" "))

    without_alphabetic = "".join(lines)
    lines = []

    for line in without_alphabetic.splitlines(keepends=True):
        without_numbers = re.sub(numeric_regex, "", line)
        lines.append(without_numbers.strip(" -"))

    without_numbers = "".join(lines)
    lines = []

    return without_numbers


@transformer
def remove_extra_whitespaces(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(re.sub(re.compile(r" {2,}"), " ", line.strip()))

    return "\n".join(lines)


def is_valid_word(s: str) -> bool:
    return all(part.isalnum() and not part.isnumeric() for part in s.split("-"))


def intersperse(value: T, sequence: Iterable[T]):
    """
    Adds value between all elements of sequence
    """

    for i, item in enumerate(sequence):
        if i != 0:
            yield value
        yield item
