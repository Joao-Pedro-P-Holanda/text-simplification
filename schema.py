from typing import Literal, TypedDict
from pydantic import BaseModel, SecretStr, computed_field

ModelOptions = Literal["cow/gemma2_tools:2b", "gemini-2.5-flash-preview-04-17"]


class Config(TypedDict):
    models: set[ModelOptions]
    llm_api_key: SecretStr
    gemini_api_key: SecretStr
    gemini_base_url: str
    llm_url: str
    notices_url: str


class ChatMessageRequestBody(BaseModel):
    text: str


class ChatMessageResponseBody(BaseModel):
    text: str


class DocumentStatistics(BaseModel):
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


class MorfologyMetrics(BaseModel): ...


class SyntaxMetrics(BaseModel): ...


class SemanticMetrics(BaseModel): ...
