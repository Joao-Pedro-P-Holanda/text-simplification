from typing import Literal, TypedDict

from pydantic import BaseModel, SecretStr, computed_field
from pathlib import Path
import re

ModelOptions = Literal[
    # Modelos Openweb UI
    "cow/gemma2_tools:2b",
    "phi4:latest",
    "phi3:latest",
    "llama3.2:latest",
    "gemma3:4b",
    "qwen2.5:14b",
    "deepseek-r1:14b",
    "granite3-dense:2b",
    "granite3-dense:8b",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-05-06",
    # Modelos OpenRouter
    "google/gemma-3n-e4b-it",
    "microsoft/phi-4",
    "meta-llama/llama-3.2-3b-instruct",
    "qwen/qwen3-14b",
    "ibm-granite/granite-4.0-h-micro",
    "deepseek/deepseek-v3.2",
    "mistralai/ministral-8b",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
]

DocumentType = Literal[
    "reference-complete", "reference-simplified", "generated-simplified"
]

TaskType = Literal[
    "readability-indexes", "nilc-metrix", "embedding-similarity", "ud-pipe"
]


class Config(TypedDict):
    llm_api_key: SecretStr
    llm_url: str


class Document(BaseModel):
    path: str
    text: str
    langchain_documents: list[str] = []

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

    def __repr__(self):
        return f"{self.name} ({self.path}"


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
    long_sentence_ratio: float

    foreign_word_ratio: float
