from typing import Literal, TypedDict
from pydantic import BaseModel, SecretStr, computed_field
from pathlib import Path

ModelOptions = Literal["cow/gemma2_tools:2b", "gemini-2.5-flash-preview-04-17"]
DocumentType = Literal[
    "reference_complete", "reference_simplified", "generated_simplified"
]


class Config(TypedDict):
    models: set[ModelOptions]
    llm_api_key: SecretStr
    llm_url: str


class Document(BaseModel):
    path: str
    text: str

    @computed_field
    @property
    def name(self) -> str:
        return Path(self.path).stem

    @computed_field
    @property
    def id(self) -> int:
        splits = self.name.split("_")

        return int(splits[0])


class DocumentResultModel(BaseModel):
    id: int
    name: str
    model: str


class DSARIMetrics(DocumentResultModel):
    f_keep: float
    f_add: float
    p_del: float

    d_keep: float
    d_add: float
    d_del: float

    sari: float
    d_sari: float


class NILCMetrics(DocumentResultModel):
    sentences_per_paragraph: float
    passive_ratio: float
    postponed_subject_ratio: float
    non_svo_ratio: float
    sentences_with_one_clause: float
    sentences_with_seven_more_clauses: float
    words_before_main_verb: float
    content_word_max: float
    content_word_min: float
    function_words: float
    ratio_function_to_content_words: float
    adjectives_ambiguity: float
    adverbs_ambiguity: float
    nouns_ambiguity: float
    verbs_ambiguity: float
    content_words_ambiguity: float
    simple_word_ratio: float
    coreference_pronoun_ratio: float
    demonstrative_pronoun_ratio: float


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
    def flesh_kincaid(self) -> float:
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
    def average_sentence_length(self) -> float:
        return self.number_of_tokens / (self.number_of_sentences or 1)

    @computed_field
    @property
    def average_sylables_word(self) -> float:
        return self.number_of_syllables / self.number_of_tokens
