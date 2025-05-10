from collections import Counter
from functools import lru_cache
import pickle
from gloe import partial_transformer
import spacy

from schema import DocumentStatistics
from utils import is_valid_word


@partial_transformer
def extract_document_statistics(
    text_data: tuple[str, set[str]],
    nlp: spacy.language.Language,
) -> DocumentStatistics:
    text, complex_words = text_data

    doc = nlp(text)

    sentences = list(doc.sents)

    num_characters = len([char for char in text if char.isalpha()])

    num_sentences = len(sentences)

    max_sentence_length = max([len(sentence) for sentence in sentences])

    tokens = [token.text.lower() for token in doc if token.is_alpha]
    # lemmas = [token.lemma_ for token in doc if token.is_alpha]
    num_syllables = sum([token._.syllables_count for token in doc if token.is_alpha])
    types = set(tokens)

    token_frequencies = Counter(tokens)

    hapax_legomena = len([count for count in token_frequencies.values() if count == 1])

    return DocumentStatistics(
        number_of_sentences=num_sentences,
        number_of_tokens=len(tokens),
        number_of_types=len(types),
        number_of_characters=num_characters,
        number_of_complex_words=len(complex_words),
        hapax_legomena_count=hapax_legomena,
        max_sentence_length=max_sentence_length,
        number_of_syllables=num_syllables,
    )


@partial_transformer
@lru_cache
def list_complex_words(text: str, frequencies_file: str) -> tuple[str, set[str]]:
    """
    Finds all the words that have a frequency lower than the set threshold in the selected corpus
    """
    result_words: set[str] = set()
    with open(frequencies_file, "rb") as file:
        frequencies = pickle.load(file)
    for word in text.split(" "):
        if is_valid_word(word) and word.lower() in frequencies:
            result_words.add(word)

    return text, result_words
