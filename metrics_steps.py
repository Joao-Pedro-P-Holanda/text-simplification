from itertools import groupby
from operator import attrgetter
import pickle
from collections import Counter
import re
from typing import Callable
import logging
import httpx
from urllib.parse import urljoin

import spacy
from gloe import partial_transformer, transformer
from spacy.matcher import Matcher
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

from schema import (
    Document,
    DocumentStatistics,
    NILCMetrics,
)
from settings import config
from utils import is_valid_word


logger = logging.getLogger(__name__)


@transformer
def transform_document_to_metric_operations(document: Document):
    return document.model_copy(
        update={"text": re.sub(re.compile(r"\*|-{2,}"), "", document.text)}
    )


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

    num_characters = sum([len(token) for token in spacy_doc if not token.is_space])

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
def extract_min_nilc_metrix(document: Document):
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
def extract_syntatic_nilc_metrix(
    document: Document,
    nlp_loader: Callable[[str], spacy.language.Language],
    model_name: str,
):
    """
    Extracts another subset of nilc metrix that depends on a licensed parser with spacy
    instead
    """

    nlp = nlp_loader(model_name)

    matcher = Matcher(nlp.vocab)

    spacy_doc = nlp(document.text)

    sentences = list(spacy_doc.sents)

    _passive_ratio(spacy_doc, len(sentences), matcher)

    _non_svo(spacy_doc)

    print(_long_sentence_ratio(sentences))


def _non_svo(doc: Doc) -> float:
    """
    Finds all sentences that are out of SVO order
    Based on the implementation in
    https://github.com/NSchrading/intro-spacy-nlp/blob/master/subject_object_extraction.py
    """

    non_svo = []

    subject_deps = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
    object_deps = ["dobj", "dative", "attr", "oprd"]

    for sent in doc.sents:
        verbs = [
            token for token in sent if token.pos_ == "VERB" and token.dep_ == "ROOT"
        ]

        for verb in verbs:
            # finds the position of the entities
            # if a subject comes after or the object comes before the sentences is not svo
            for child in verb.children:
                if child.dep_ in subject_deps and child.dep > verb.i:
                    non_svo.append(verb.sent)
                elif child.dep_ in object_deps and child.i < verb.i:
                    non_svo.append(verb.sent)

    return len(non_svo) / len(list(doc.sents))


def _passive_ratio(doc, num_of_sents: int, matcher: Matcher):
    passive_rule = [{"DEP": "nsubjpass"}]
    matcher.add("Passive voice", [passive_rule])

    result = len(matcher(doc)) / num_of_sents

    return result


def _words_before_main_verb(sents: list[Span]) -> float: ...


def _long_sentence_ratio(sents: list[Span]) -> float:
    """
    Proportion of sentences with more than 15 words
    """
    i = 0
    for sent in sents:
        tokens = [token.text.lower() for token in sent if token.is_alpha]
        if len(tokens) > 15:
            i += 1

    return i / len(sents)


def _aux_plus_PCP_per_sentence(sents: list[Span]) -> float: ...


def _coreference_pronoun_ratio(sents: list[Span]) -> float: ...


def _demonstrative_pronoun_ratio(sents: list[Span]) -> float: ...


def _foreign_word_ratio(sents: list[Span]) -> float: ...


@transformer
def group_documents_by_model(documents: list[Document]) -> list[list[Document]]:
    by_name = groupby(documents, key=attrgetter("generated_with"))
    result = [list(group[1]) for group in by_name]

    return result


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
