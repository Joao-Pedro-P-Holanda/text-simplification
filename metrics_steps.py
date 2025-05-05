from collections import Counter
from gloe import partial_transformer
import spacy

from schema import StatisticsMetrics


@partial_transformer
def extract_sentence_statistics(
    text: str,
    nlp: spacy.language.Language,
) -> StatisticsMetrics:
    doc = nlp(text)

    sentences = list(doc.sents)

    total_sentences = len(sentences)

    tokens = [token.text.lower() for token in doc if token.is_alpha]

    avg_sentence_length = len(tokens) / (total_sentences or 1)

    token_frequencies = Counter(tokens)

    print(token_frequencies)

    hapax_legomena = len([count for count in token_frequencies.values() if count == 1])

    return StatisticsMetrics(
        number_of_sentences=total_sentences,
        average_sentence_length=avg_sentence_length,
        hapax_legomena_count=hapax_legomena,
    )
