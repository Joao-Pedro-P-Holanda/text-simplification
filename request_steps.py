from functools import lru_cache
import logging

from gloe import partial_transformer
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.messages.ai import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM

from schema import Document, ModelOptions
from settings import config

logger = logging.getLogger(__name__)


@partial_transformer
def request_simplfied_text_from_chat_model(
    input: tuple[Document, ModelOptions], prompt_file: str
) -> tuple[Document, str]:
    document, model = input

    llm_chain = _read_prompt_file(prompt_file) | _llm_for_model_name(model)

    logger.info(f"Requesting API for simplified text of file {document.name}")
    response = llm_chain.invoke({"text": document.text})

    text = response.text() if isinstance(response, AIMessage) else response

    return document.model_copy(update={"text": text}), model


@lru_cache
def _read_prompt_file(prompt_file: str) -> PromptTemplate:
    with open(prompt_file, "r") as f:
        return PromptTemplate.from_template("".join(f.readlines()))


def _llm_for_model_name(model: ModelOptions) -> BaseLLM | BaseChatModel:
    match model:
        case "gemini-2.5-flash-preview-04-17":
            return ChatGoogleGenerativeAI(
                model=model, temperature=0, max_tokens=None, timeout=None, max_retries=1
            )
        case "cow/gemma2_tools:2b":
            headers = {
                "Authorization": f"Bearer {config['llm_api_key']}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            return OllamaLLM(
                model=model,
                temperature=0,
                base_url=config["llm_url"],
                client_kwargs={"headers": headers, "timeout": 360},
            )
        case _:
            raise ValueError(f"Can't find implementation for model {model}")
