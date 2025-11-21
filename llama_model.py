from typing import override
from langchain_ollama import ChatOllama
from deepeval.models.base_model import DeepEvalBaseLLM


class OllamaDeepeval(DeepEvalBaseLLM):
    def __init__(self, model: ChatOllama):
        self.model: ChatOllama = model

    @override
    def load_model(self):
        return self.model

    @override
    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    @override
    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    @override
    def get_model_name(self) -> str:
        return self.model.model
