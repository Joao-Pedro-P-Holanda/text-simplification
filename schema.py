from typing import Literal, TypedDict
from pydantic import BaseModel, SecretStr

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


class StatisticsMetrics(BaseModel):
    number_of_sentences: int
    average_sentence_length: float
    hapax_legomena_count: int


class MorfologyMetrics(BaseModel): ...


class SyntaxMetrics(BaseModel): ...


class SemanticMetrics(BaseModel): ...
