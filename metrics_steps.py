import csv
import logging
import os
import pickle
import re
import string
from itertools import chain, groupby, pairwise
from operator import attrgetter
from statistics import mean
from typing import Callable

from gloe import partial_transformer, transformer

from conllu_adapted import Conllu, Token
from schema import (
    Document,
    DocumentStatistics,
    UDNilcMetrics,
)
from utils import (
    is_valid_word,
    remove_enumerations,
    remove_extra_whitespaces,
    remove_markdown_tables,
    strip_special_characters,
)

logger = logging.getLogger(__name__)


@transformer
def remove_duplicate_captions(document: Document) -> Document:
    """
    Remove text from url captions in the format !?[<caption>](url)
    that is duplicated in the text
    """

    url_captions = re.compile(r"\[(.*?)\]\(.*?\)").findall(document.text)
    for caption in url_captions:
        escaped_caption = re.escape(caption)
        # using small number to avoid wrongly parsed links, such as values in
        # the end of lines
        if len(escaped_caption) > 10:
            re.sub(rf"(?<!\[){escaped_caption}(?!\])", "", document.text)
    return document


@transformer
def transform_document_to_metric_operations(document: Document) -> Document:
    new_text = (
        remove_markdown_tables
        >> strip_special_characters
        >> remove_extra_whitespaces
        >> remove_enumerations
        >> remove_extra_whitespaces
    )(document.text)

    return document.model_copy(update={"text": new_text})


@transformer
def extract_document_statistics_port_parser(
    text_data: tuple[Document, set[str]],
) -> DocumentStatistics:
    document, complex_words = text_data

    logger.info(f"Extracting readability indexes from document {document.name}")
    try:
        conllu = Conllu.from_file(document.path.replace(".md", ".conllu"))

        return DocumentStatistics(
            id=document.id,
            name=document.name,
            year=document.year,
            generated_with=document.generated_with,
            model="port_parser",
            max_sentence_length=conllu.max_sentence_length,
            number_of_sentences=len(conllu.sentences),
            number_of_complex_words=len(complex_words),
            hapax_legomena_count=len(conllu.hapax_legomena),
            number_of_types=len(conllu.types),
            number_of_tokens=conllu.token_count,
            number_of_syllables=conllu.syllables_count_pyphen,
            number_of_characters=conllu.character_count,
        )

    except:
        raise


@transformer
def extract_min_nilc_metrix_ud(document: Document) -> UDNilcMetrics:
    """
    Extracts the following metrics from NILC using a conllu file:

    - Non-SVO Sentences Ratio
    - Passive Voice Ratio
    - Long Sentence (over 20 words) Ratio
    - Words before main verb
    - Personal Pronoun Ratio
    - Coreference Pronoun Ratio
    - Demonstrative Pronoun Ratio
    - Logical Operator Ratio
    - Connective Ratio

    The following metric is also extracted, albeit not being present on the NILC Metrix project.
    - Foreign Word Ratio
    """

    # TODO: keep decontracted tokens, but don't use in word counts

    conllu = Conllu.from_file(document.path.replace(".md", ".conllu"))

    return UDNilcMetrics(
        id=document.id,
        name=document.name,
        year=document.year,
        model="portuguese-porttinari-ud-2.15-241121",
        generated_with=document.generated_with,
        long_sentence_ratio=_long_sentence_ratio_ud(conllu),
        non_svo_ratio=_non_svo_ud(conllu),
        passive_voice_ratio=_passive_ratio_ud(conllu),
        words_before_main_verb_mean=_words_before_main_verb_ud(conllu),
        personal_pronoun_ratio=_personal_pronoun_ratio_ud(conllu),
        coreference_pronoun_ratio=_coreference_pronoun_ratio_ud(conllu),
        demonstrative_pronoun_ratio=_demonstrative_pronoun_ratio_ud(conllu),
        foreign_word_ratio=_foreign_word_ratio(conllu),
    )


def _non_svo_ud(conllu: Conllu):
    subject_deps = [
        "nsubj",
        "nsubj:pass",
        "nsubj:outer",
        "csubj",
        "csubj:pass",
        "csubj:outer",
    ]
    object_deps = ["obj"]

    non_svo_sentences = []
    for sentence in conllu.sentences:
        subject_position = -1
        verb_position = -1
        object_position = -1
        for token in sentence.tokens:
            if token.deprel in subject_deps and subject_position == -1:
                subject_position = (
                    min(token.id) if isinstance(token.id, tuple) else token.id
                )
            elif (
                token.upos == "VERB" and token.deprel == "root" and verb_position == -1
            ):
                verb_position = (
                    min(token.id) if isinstance(token.id, tuple) else token.id
                )
            elif token.deprel in object_deps and object_position == -1:
                object_position = (
                    min(token.id) if isinstance(token.id, tuple) else token.id
                )

            if object_position != -1 and subject_position != -1 and verb_position != -1:
                if object_position < verb_position or subject_position > verb_position:
                    non_svo_sentences.append(sentence)
                break

    return len(non_svo_sentences) / len(conllu.sentences)


def _passive_ratio_ud(conllu: Conllu):
    """
    Returns the ratio of sentences that are in the passive voice

    Currently, the PortParser trained model couldn't find synthetic passive voice with
    passive particles: "Vendem-se peixes"
    but the dependency relation expl:pass is a valid UD relation.
    """

    passive_rule = ["csubj:pass", "nsubj:pass", "aux:pass", "expl:pass", "obl:agent"]

    passive_voice_sentences = []
    for sentence in conllu.sentences:
        for token in sentence.tokens:
            if token.deprel in passive_rule:
                passive_voice_sentences.append(sentence)
                break

    return len(passive_voice_sentences) / len(conllu.sentences)


def _words_before_main_verb_ud(conllu: Conllu) -> float:
    words_before_main_verb_per_sentence = []
    for sentence in conllu.sentences:
        for idx, token in enumerate(sentence.tokens):
            if token.upos == "VERB" and token.deprel == "root":
                words_before_main_verb_per_sentence.append(idx)
                break

    return mean(words_before_main_verb_per_sentence)


def _long_sentence_ratio_ud(conllu: Conllu) -> float:
    long_sentences = []
    for sentence in conllu.sentences:
        words = list(
            filter(lambda tok: tok.form not in string.punctuation, sentence.tokens)
        )

        if len(words) > 20:
            long_sentences.append(words)

    return len(long_sentences) / len(conllu.sentences)


def _personal_pronoun_ratio_ud(conllu: Conllu) -> float:
    tokens = list(
        chain.from_iterable([sentence.tokens for sentence in conllu.sentences])
    )

    personal = list(
        filter(
            lambda tok: tok.upos == "PRON"
            and tok.feats
            and tok.feats.get("PronType") == "Prs",
            tokens,
        )
    )

    return len(personal) / (len(tokens) or 1)


def _coreference_pronoun_ratio_ud(conllu: Conllu) -> float:
    """
    Number of possible referents in the direct previous sentence based on the subject pronouns
    (ele, ela, elas, eles) of the current sentence.
    """

    subject_pronouns = ["ele", "ela"]

    subject_pronouns_exp = lambda tok: tok.lemma in subject_pronouns

    return _pronoun_coreference_average(conllu, subject_pronouns_exp)


def _demonstrative_pronoun_ratio_ud(conllu: Conllu) -> float:
    """
    Number of possible referents in the direct previous sentence based on the demonstratives pronouns
    """
    demonstrative_pronouns_exp = (
        lambda tok: tok.feats and tok.feats.get("PronType") == "Dem"
    )

    return _pronoun_coreference_average(conllu, demonstrative_pronouns_exp)

def _foreign_word_ratio(conllu: Conllu, write: bool = True) -> float:
    foreign_words = set()

    tokens = list(
        chain.from_iterable([sentence.tokens for sentence in conllu.sentences])
    )

    for token in tokens:
        if (
            token.feats and token.feats.get("Foreign") == "Yes"
        ) or token.deprel == "flat:foreign":
            foreign_words.add(token.lemma)

    store_path = "foreign_words_predicted.csv"

    if write:
        number_of_lines = 0
        if os.path.exists(store_path):
            with open(store_path, "r") as f:
                number_of_lines = len(f.readlines())

        with open(store_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=None)  # type:ignore

            writer.fieldnames = ["name", "word", "word_count"]
            if number_of_lines == 0:
                writer.writeheader()

            for word in foreign_words:
                data = {
                    "name": conllu.path.stem.replace("_stripped.predicted", ""),
                    "word": word,
                    "word_count": len(tokens)
                }
                writer.writerow(data)

    return len(foreign_words) / (len(conllu.lemmas) or 1)


def _pronoun_coreference_average(
    conllu: Conllu, pronouns_exp: Callable[[Token], bool]
) -> float:
    """
    Extracts the number of possible coreferences for the given pronoun filter in the
    pairwise sentences, a coreference occurs when the pronoun has gender and number
    concordance with a token on the previous sentence
    """
    mean_referents_for_sentence = []

    for previous_sentence, current_sentence in pairwise(conllu.sentences):
        subject_pronouns = list(
            filter(
                lambda tok: pronouns_exp(tok),
                current_sentence.tokens,
            )
        )

        coreference_classes = ["PROPN", "NOUN"]

        possible_coreferenced = []
        for pronoun in subject_pronouns:
            for token in filter(
                lambda tok: tok.upos in coreference_classes,
                previous_sentence.tokens,
            ):
                gender = token.feats.get("Gender") if token.feats else None
                number = token.feats.get("Number") if token.feats else None

                if (
                    (gender and number)
                    and gender == pronoun.feats.get("Gender")
                    and number == pronoun.feats.get("Number")
                ):
                    possible_coreferenced.append(token)

        if subject_pronouns:
            mean_referents_for_sentence.append(
                len(possible_coreferenced) / (len(subject_pronouns) or 1)
            )

    return mean(mean_referents_for_sentence) if mean_referents_for_sentence else 0


@transformer
def group_documents_by_model(documents: list[Document]) -> list[list[Document]]:
    by_name = groupby(documents, key=attrgetter("generated_with"))
    result = [list(group[1]) for group in by_name]

    return result


@partial_transformer
def list_complex_words(
    document: Document,
    frequencies_file: str,
) -> tuple[Document, set[str]]:
    """
    Finds all the words that have a frequency lower than the set threshold in the selected corpus
    """
    # TODO: use lemmas instead of the raw ocurrences
    logger.info(f"Finding complex words on {document.name}")

    conllu = Conllu.from_file(document.path.replace(".md", ".conllu"))

    tokens = list(
        chain.from_iterable([sentence.tokens for sentence in conllu.sentences])
    )

    if any([token.lemma is None for token in tokens]):
        print(f"{document.path} has none lemmas")

    # compound words such as 'do (de+o) don't have their lemma defined

    tokens = list(filter(lambda tok: is_valid_word(tok.lemma or tok.form), tokens))

    words = [tok.lemma or tok.form for tok in tokens]

    result_words: set[str] = set()
    with open(frequencies_file, "rb") as file:
        frequencies = pickle.load(file)
    for word in words:
        if word.lower() not in frequencies:
            result_words.add(word)

    return document, result_words
