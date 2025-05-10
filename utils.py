def is_valid_word(s: str) -> bool:
    return all(part.isalpha() for part in s.split("-"))
