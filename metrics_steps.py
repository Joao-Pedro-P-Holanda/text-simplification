import logging
import pickle
import re
import string
from collections import Counter
from itertools import chain, groupby, pairwise
from operator import attrgetter
from statistics import mean
from typing import Callable, Literal, overload
from urllib.parse import urljoin

import httpx
import spacy
from gloe import partial_transformer, transformer
from spacy.matcher import Matcher
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

from conllu_adapted import Conllu, Token
from schema import (
    Document,
    DocumentStatistics,
    NILCMetrics,
    UDNilcMetrics,
)
from settings import config
from utils import (
    is_valid_word,
    remove_enumerations,
    remove_extra_whitespaces,
    remove_markdown_tables,
    strip_special_characters,
)

logger = logging.getLogger(__name__)


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

    tokens = [token.text for token in spacy_doc if token.is_alpha]
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
        logical_operator_ratio=_logical_operator_ud(conllu),
        connective_ratio=_connective_ratio_ud(conllu),
        foreign_word_ratio=_foreign_word_ratio(conllu),
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

    print(_passive_ratio(spacy_doc, len(sentences), matcher))

    print(_non_svo(spacy_doc))

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

    Currently, the porttinarit trained model couldn't find synthetic passive voice with
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


def _passive_ratio(doc, num_of_sents: int, matcher: Matcher):
    passive_rule = [{"DEP": "nsubjpass"}]
    matcher.add("Passive voice", [passive_rule])

    result = len(matcher(doc)) / num_of_sents

    return result


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


def _logical_operator_ud(conllu: Conllu) -> float:
    logical_operators = [
        r"\We\W",
        r"\Wou\W",
        r"\Wse\W",
        r"\Wnão\W",
        r"\Wnem\W",
        r"\Wnenhum\W",
        r"\Wnenhuma\W",
        r"\Wnada\W",
        r"\Wnunca\W",
        r"\Wjamais\W",
        r"\Wcaso\W",
        r"\Wdesde que\W",
        r"\Wcontanto que\W",
        r"\Wuma vez que\W",
        r"\Wa menos que\W",
        r"\Wsem que\W",
        r"\Wa não ser que\W",
        r"\Wsalvo se\W",
        r"\Wexceto se\W",
        r"\Wentão é porque\W",
        r"\Wvai que\W",
        r"\Wvá que\W",
    ]

    occurrences: list[tuple[int, int]] = []
    words = list(
        filter(
            lambda token: not token.form.isnumeric()
            and token.form not in string.punctuation,
            chain.from_iterable([sentence.tokens for sentence in conllu.sentences]),
        )
    )

    for logical_ops in logical_operators:
        matched_spans = re.finditer(logical_ops, conllu.text, re.IGNORECASE)
        for match in matched_spans:
            if not any(
                [
                    match.start() > occurrence[0] and match.start() < occurrence[1]
                    for occurrence in occurrences
                ]
            ):
                occurrences.append((match.start(), match.end()))

    return len(occurrences) / len(words)


def _connective_ratio_ud(conllu: Conllu) -> float:
    # TODO: change to a less inneficient method

    connectives = [
        r"\Wna realidade\W",
        r"\Wem resumo\W",
        r"\Wcomo resultado\W",
        r"\Wpelo menos\W",
        r"\Walém disso\W",
        r"\Walém disto\W",
        r"\Wdevido a\W",
        r"\Wdevido à\W",
        r"\Whabilita\W",
        r"\Wmesmo que\W",
        r"\Wsegue que\W",
        r"\Wpara\W",
        r"\Wfelizmente\W",
        r"\Walém de\W",
        r"\Walém da\W",
        r"\Walém do\W",
        r"\Wse\W",
        r"\Wna verdade\W",
        r"\Wde qualquer modo\W",
        r"\Wde qualquer maneira\W",
        r"\Wem conclusão\W",
        r"\Wem outras palavras\W",
        r"\Wem vez de\W",
        r"\Wsimilarmente\W",
        r"\Wpor outro lado\W",
        r"\Wde novo\W",
        r"\Wsomente se\W",
        r"\Wdado que\W",
        r"\Wdesde que\W",
        r"\Wassim\W",
        r"\Wpara dar um exemplo\W",
        r"\Wpara este fim\W",
        r"\Wde qualquer forma\W",
        r"\Walternativamente\W",
        r"\Wao acaso\W",
        r"\Wem todo caso\W",
        r"\Wmesmo embora\W",
        r"\Wentretanto\W",
        r"\Wcontudo\W",
        r"\Wno entanto\W",
        r"\Wapesar de\W",
        r"\Wapesar disso\W",
        r"\Wapesar disto\W",
        r"\Wnem\W",
        r"\Wpelo contrário\W",
        r"\Wou então\W",
        r"\Wde outra forma\W",
        r"\Wde outro modo\W",
        r"\Wde outra maneira\W",
        r"\Wao menos que\W",
        r"\Wa menos que\W",
        r"\Wainda\W",
        r"\Wnovamente\W",
        r"\Woutra vez\W",
        r"\Wimediatamente\W",
        r"\Wnesse ponto\W",
        r"\Wneste ponto\W",
        r"\Wnesse momento\W",
        r"\Wneste momento\W",
        r"\Wmais longe\W",
        r"\Wmais distante\W",
        r"\Wlogo\W",
        r"\Wum momento antes de\W",
        r"\Wtardio\W",
        r"\Wagora que\W",
        r"\Wem outra ocasião\W",
        r"\Wem outro momento\W",
        r"\Wem outra hora\W",
        r"\Wem outra vez\W",
        r"\Wdaqui a pouco\W",
        r"\Wpreviamente\W",
        r"\Wsimultaneamente\W",
        r"\Waté agora\W",
        r"\Waté aqui\W",
        r"\Wem breve\W",
        r"\Wsubitamente\W",
        r"\Wrepentinamente\W",
        r"\Wúltima vez\W",
        r"\Wdurante todo o tempo\W",
        r"\Wapenas\W",
        r"\Wmal\W",
        r"\Wprimeiro que\W",
        r"\Wtodas as vezes que\W",
        r"\Wao passo que\W",
        r"\Wbem como\W",
        r"\Wmais tarde\W",
        r"\Wdaí\W",
        r"\Wdessa forma\W",
        r"\Wé claro\W",
        r"\Wentão\W",
        r"\Wnesse caso\W",
        r"\Wneste caso\W",
        r"\Wpor conseguinte\W",
        r"\Wpor esta razão\W",
        r"\Wpor essa razão\W",
        r"\Wpor este motivo\W",
        r"\Wpor esse motivo\W",
        r"\Wafinal de contas\W",
        r"\Wcondicional a\W",
        r"\Wcondicional à\W",
        r"\Wquer dizer que\W",
        r"\Wna condição de\W",
        r"\Wpropósito de\W",
        r"\Wpropósito da\W",
        r"\Wpropósito do\W",
        r"\Wrelativo a\W",
        r"\Wrelativo à\W",
        r"\Wdesde\W",
        r"\Wsempre que\W",
        r"\Wcada vez que\W",
        r"\Wembora\W",
        r"\Wa propósito\W",
        r"\Wde fato\W",
        r"\Wincidentalmente\W",
        r"\Wem segundo lugar\W",
        r"\Wsegue\W",
        r"\Wem contraste\W",
        r"\Wadicionalmente\W",
        r"\Wao invés de\W",
        r"\Wao mesmo tempo\W",
        r"\Wcomo\W",
        r"\Wpor exemplo\W",
        r"\Wdesse modo\W",
        r"\We\W",
        r"\Wem adição\W",
        r"\Wem geral\W",
        r"\Wem paralelo\W",
        r"\Wincluse\W",
        r"\Wisto é\W",
        r"\Wou seja\W",
        r"\Wpor fim\W",
        r"\Wresumindo\W",
        r"\Wtambém\W",
        r"\Wfinalmente\W",
        r"\Wmas\W",
        r"\Wporém\W",
        r"\Wtodavia\W",
        r"\Wantes\W",
        r"\Wcom\W",
        r"\Wpois\W",
        r"\Wpor causa\W",
        r"\Wpor isso\W",
        r"\Wporque\W",
        r"\Wportanto\W",
        r"\Wsendo assim\W",
        r"\Wnesse contexto\W",
        r"\Wnesse sentido\W",
        r"\Wvisto isso\W",
        r"\Wa partir de\W",
        r"\Wmesmo assim\W",
        r"\Watravés\W",
        r"\Wuma vez que\W",
        r"\Wa fim de\W",
        r"\Waí\W",
        r"\Wconsequentemente\W",
        r"\Wdessa maneira\W",
        r"\Wenquanto\W",
        r"\Wprimeiramente\W",
        r"\Wa seguir\W",
        r"\Wposteriormente\W",
        r"\Wapós\W",
        r"\Wdepois\W",
        r"\Wcaso\W",
        r"\Wexceto que\W",
        r"\Wa medida que\W",
        r"\Watualmente\W",
        r"\Wquando\W",
        r"\Wem seguida\W",
        r"\Wlogo após\W",
        r"\Wjá\W",
        r"\Waté que\W",
    ]

    occurrences: list[tuple[int, int]] = []
    words = list(
        filter(
            lambda token: not token.form.isnumeric()
            and token.form not in string.punctuation,
            chain.from_iterable([sentence.tokens for sentence in conllu.sentences]),
        )
    )

    for connective in connectives:
        matched_spans = re.finditer(connective, conllu.text, re.IGNORECASE)
        for match in matched_spans:
            if not any(
                [
                    match.start() > occurrence[0] and match.start() < occurrence[1]
                    for occurrence in occurrences
                ]
            ):
                occurrences.append((match.start(), match.end()))

    return len(occurrences) / len(words)


def _foreign_word_ratio(conllu: Conllu) -> float:
    foreign_words = set()

    tokens = list(
        chain.from_iterable([sentence.tokens for sentence in conllu.sentences])
    )

    for token in tokens:
        if token.feats and token.feats.get("Foreign") == "Yes":
            foreign_words.add(token.form.lower())

    return len(foreign_words) / (len(conllu.types) or 1)


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
    tokenizer: Literal["spacy", "udpipe"],
    model_name: str | None = None,
    nlp_loader: Callable[[str], spacy.language.Language] | None = None,
) -> tuple[Document, set[str]]:
    """
    Finds all the words that have a frequency lower than the set threshold in the selected corpus
    """
    # TODO: use lemmas instead of the raw ocurrences
    logger.info(f"Finding complex words on {document.name}")

    if tokenizer == "spacy":
        nlp = nlp_loader(model_name)

        spacy_doc = nlp(document.text)

        words = [token.lemma_ for token in spacy_doc if is_valid_word(token.lemma_)]

    elif tokenizer == "udpipe":
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
