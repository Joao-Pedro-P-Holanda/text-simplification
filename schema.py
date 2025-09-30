from typing import Literal, TypedDict

from langchain_core.documents import Document as LangchainDocument
from pydantic import BaseModel, SecretStr, computed_field, HttpUrl
from pathlib import Path
import re

ModelOptions = Literal[
    "cow/gemma2_tools:2b",
    "phi4:latest",
    "phi3:latest",
    "llama3.2:latest",
    "gemma3:4b",
    "qwen2.5:14b",
    "qwen2.5-coder:32b",
    "deepseek-r1:14b",
    "granite3-dense:2b",
    "granite3-dense:8b",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-05-06",
]

EmbeddingModelOptions = Literal["nomic-embed-text-v2", "nomic-embed-text-v1.5"]

DocumentType = Literal[
    "reference-complete", "reference-simplified", "generated-simplified"
]

TaskType = Literal[
    "readability-indexes", "d-sari", "nilc-metrix", "embedding-similarity", "ud-pipe"
]


class Config(TypedDict):
    llm_api_key: SecretStr
    llm_url: str
    nilc_metrix_url: HttpUrl


class Document(BaseModel):
    path: str
    text: str
    langchain_documents: list[LangchainDocument] = []

    @computed_field
    @property
    def generated_with(self) -> ModelOptions | None:
        parts = Path(self.path).parts
        pattern = re.compile(r"/|:|-")
        names = {re.sub(pattern, "", arg): arg for arg in ModelOptions.__args__}
        for part in parts:
            key = re.sub(pattern, "", str(part))
            if key in names:
                return names[key]
        return None

    @computed_field
    @property
    def name(self) -> str:
        return Path(self.path).stem

    @computed_field
    @property
    def id(self) -> int:
        splits = self.name.split("_")

        for i in splits[::-1]:
            try:
                return int(i)
            except:
                pass

    @computed_field
    @property
    def year(self) -> int:
        splits = self.name.split("_")

        return int(splits[0])


class DocumentResultModel(BaseModel):
    id: int
    name: str
    year: int
    model: str | None = None
    generated_with: ModelOptions | None


class EmbeddingCosineSimilarity(DocumentResultModel):
    id: int
    name: str
    model: str
    original_simplified_similarity: float


class DSARIMetrics(DocumentResultModel):
    f_keep: float
    f_add: float
    p_del: float

    d_keep: float
    d_add: float
    d_del: float

    sari: float
    d_sari: float


class DocumentStatistics(DocumentResultModel):
    """
    This class describes the most used readability techniques with their constants
    modified to better suit Brazilian Portuguese, as describe in the alt system.
    doi: 10.61358/policromias.v8i1.54352
    """

    number_of_sentences: int
    number_of_tokens: int
    number_of_types: int
    number_of_characters: int
    number_of_syllables: float
    number_of_complex_words: int
    hapax_legomena_count: int
    max_sentence_length: int

    @computed_field
    @property
    def flesch_ease(self) -> float:
        return (
            226
            - 1.04 * (self.number_of_tokens / (self.number_of_sentences or 1))
            - 72 * (self.number_of_syllables / (self.number_of_tokens))
        )

    @computed_field
    @property
    def flesch_kincaid(self) -> float:
        return (
            0.36 * (self.number_of_tokens / (self.number_of_sentences or 1))
            + 10.4 * (self.number_of_syllables / (self.number_of_tokens or 1))
            - 18
        )

    @computed_field
    @property
    def ari(self) -> float:
        return (
            4.6 * (self.number_of_characters / (self.number_of_tokens or 1))
            + 0.44 * (self.number_of_tokens / (self.number_of_sentences or 1))
            - 20
        )

    @computed_field
    @property
    def gunning_fog(self) -> float:
        return 0.49 * (self.number_of_tokens / (self.number_of_sentences or 1)) + 19 * (
            self.number_of_complex_words / (self.number_of_tokens or 1)
        )

    @computed_field
    @property
    def coleman_liau(self) -> float:
        return (
            5.4 * (self.number_of_characters / (self.number_of_tokens or 1))
            - 21 * (self.number_of_sentences / (self.number_of_tokens or 1))
            - 14
        )

    @computed_field
    @property
    def gulpease(self) -> float:
        return (
            89
            + 300 * (self.number_of_sentences / self.number_of_tokens)
            - 10 * (self.number_of_characters / self.number_of_tokens)
        )

    @computed_field
    @property
    def average_sentence_length(self) -> float:
        return self.number_of_tokens / (self.number_of_sentences or 1)

    @computed_field
    @property
    def average_syllables_word(self) -> float:
        return self.number_of_syllables / self.number_of_tokens


class UDNilcMetrics(DocumentResultModel):
    non_svo_ratio: float
    passive_voice_ratio: float
    words_before_main_verb_mean: float
    personal_pronoun_ratio: float
    coreference_pronoun_ratio: float
    demonstrative_pronoun_ratio: float
    logical_operator_ratio: float
    connective_ratio: float
    long_sentence_ratio: float

    foreign_word_ratio: float


class NILCMetrics(DocumentResultModel):
    sentences_per_paragraph: float  # ok
    # sentences_with_one_clause: float
    # sentences_with_seven_more_clauses: float

    function_words: float  # ok
    ratio_function_to_content_words: float  # ok
    adjectives_ambiguity: float  # ok
    adverbs_ambiguity: float  # ok
    nouns_ambiguity: float  # ok
    verbs_ambiguity: float  # ok
    # coreference_pronoun_ratio: float
    # demonstrative_pronoun_ratio: float

    adjective_ratio: float
    adverbs: float
    content_words: float
    flesch: float
    syllables_per_content_word: float
    words_per_sentence: float
    noun_ratio: float
    paragraphs: int
    sentences: int
    words: int
    pronoun_ratio: float
    verbs: float
    logic_operators: float
    and_ratio: float
    if_ratio: float
    or_ratio: float
    negation_ratio: float
    cw_freq: float
    cw_freq_brwac: float
    cw_freq_bra: float
    min_cw_freq: float
    min_cw_freq_brwac: float
    min_freq_brwac: float
    min_cw_freq_bra: float
    min_freq_bra: float
    freq_brwac: float
    freq_bra: float
    hypernyms_verbs: float
    brunet: float
    honore: float
    personal_pronouns: float
    ttr: float
    conn_ratio: float
    add_neg_conn_ratio: float
    add_pos_conn_ratio: float
    cau_neg_conn_ratio: float
    cau_pos_conn_ratio: float
    log_neg_conn_ratio: float
    log_pos_conn_ratio: float
    tmp_neg_conn_ratio: float
    tmp_pos_conn_ratio: float
    yngve: float
    frazier: float
    dep_distance: float
    cross_entropy: float
    content_density: float
    adjacent_refs: float
    anaphoric_refs: float
    adj_arg_ovl: float
    arg_ovl: float
    adj_stem_ovl: float
    stem_ovl: float
    adj_cw_ovl: float
    lsa_adj_mean: float
    lsa_adj_std: float
    lsa_all_mean: float
    lsa_all_std: float
    lsa_paragraph_mean: float
    lsa_paragraph_std: float
    lsa_givenness_mean: float
    lsa_givenness_std: float
    lsa_span_mean: float
    lsa_span_std: float
    negative_words: float
    positive_words: float
