from typing import Literal, TypedDict
from pydantic import BaseModel, SecretStr

ModelOptions = Literal["cow/gemma2_tools:2b"]


class Config(TypedDict):
    models: set[ModelOptions]
    llm_api_key: SecretStr
    llm_url: str
    notices_url: str


class ChatMessageRequestBody(BaseModel):
    text: str


class ChatMessageResponseBody(BaseModel):
    text: str


class StatisticsMetrics(BaseModel):
    number_of_sentences: int
    average_sentence_length: int
    hapax_legomena_count: int


class MorfologyMetrics(BaseModel): ...


class SyntaxMetrics(BaseModel): ...


class SemanticMetrics(BaseModel): ...
