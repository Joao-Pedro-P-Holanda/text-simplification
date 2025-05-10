import pickle
from collections import Counter

import spacy
from gloe import partial_transformer, transformer
from spacy.matcher import Matcher

from schema import Document, DocumentStatistics
from utils import is_valid_word


@partial_transformer
def extract_document_statistics(
    text_data: tuple[Document, set[str]],
    nlp: spacy.language.Language,
) -> DocumentStatistics:
    document, complex_words = text_data

    spacy_doc = nlp(document.text)

    sentences = list(spacy_doc.sents)

    num_characters = len([char for char in document.text if char.isalpha()])

    num_sentences = len(sentences)

    max_sentence_length = max([len(sentence) for sentence in sentences])

    tokens = [token.text.lower() for token in spacy_doc if token.is_alpha]
    # lemmas = [token.lemma_ for token in doc if token.is_alpha]
    num_syllables = sum(
        [token._.syllables_count for token in spacy_doc if token.is_alpha]
    )
    types = set(tokens)

    token_frequencies = Counter(tokens)

    hapax_legomena = len([count for count in token_frequencies.values() if count == 1])

    return DocumentStatistics(
        id=document.id,
        name=document.name,
        number_of_sentences=num_sentences,
        number_of_tokens=len(tokens),
        number_of_types=len(types),
        number_of_characters=num_characters,
        number_of_complex_words=len(complex_words),
        hapax_legomena_count=hapax_legomena,
        max_sentence_length=max_sentence_length,
        number_of_syllables=num_syllables,
    )


@transformer
def calculate_nilc_metrix(text: str, nlp: spacy.language.Language):
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)

    non_svo_pattern = []

    passive_pattern = []


@partial_transformer
def list_complex_words(
    document: Document, frequencies_file: str
) -> tuple[Document, set[str]]:
    """
    Finds all the words that have a frequency lower than the set threshold in the selected corpus
    """
    result_words: set[str] = set()
    with open(frequencies_file, "rb") as file:
        frequencies = pickle.load(file)
    for word in document.text.split(" "):
        if is_valid_word(word) and word.lower() in frequencies:
            result_words.add(word)

    return document, result_words
