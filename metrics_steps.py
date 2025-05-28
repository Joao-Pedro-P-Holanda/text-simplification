import pickle
from collections import Counter
from typing import Callable
import logging
import httpx
from urllib.parse import urljoin

import spacy
from gloe import partial_transformer, transformer
from spacy.matcher import Matcher

from schema import (
    Document,
    DocumentStatistics,
    NILCMetrics,
)
from settings import config
from utils import is_valid_word


logger = logging.getLogger(__name__)


@partial_transformer
def extract_document_statistics(
    text_data: tuple[Document, set[str]],
    nlp_loader: Callable[[str], spacy.language.Language],
    model_name: str,
) -> DocumentStatistics:
    document, complex_words = text_data

    logger.info(f"Extracting readability indexes from document {document.name}")

    nlp = nlp_loader(model_name)

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
        year=document.year,
        number_of_sentences=num_sentences,
        number_of_tokens=len(tokens),
        number_of_types=len(types),
        number_of_characters=num_characters,
        number_of_complex_words=len(complex_words),
        hapax_legomena_count=hapax_legomena,
        max_sentence_length=max_sentence_length,
        number_of_syllables=num_syllables,
        model=model_name,
        generated_with=document.generated_with,
    )


@transformer
def extract_nilc_metrix(document: Document):
    metrics_url = urljoin(
        str(config["nilc_metrix_url"]), "/api/v1/metrix/_min/yyy?format=json"
    )
    headers = {"Content-Type": "text"}

    logger.info(f"Requesting nilc-metrix for document {document.name} at {metrics_url}")
    response = httpx.post(
        metrics_url, headers=headers, content=document.text, timeout=None
    )

    response_dict = response.json()

    response_dict.update(document.model_dump())

    return NILCMetrics.model_validate(response_dict)


@partial_transformer
def list_complex_words(
    document: Document, frequencies_file: str
) -> tuple[Document, set[str]]:
    """
    Finds all the words that have a frequency lower than the set threshold in the selected corpus
    """
    logger.info(f"Finding complex words on {document.name}")
    result_words: set[str] = set()
    with open(frequencies_file, "rb") as file:
        frequencies = pickle.load(file)
    for word in document.text.split(" "):
        if is_valid_word(word) and word.lower() in frequencies:
            result_words.add(word)

    return document, result_words
