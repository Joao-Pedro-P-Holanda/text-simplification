from functools import lru_cache
import logging

from gloe import partial_transformer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.chat_models import ChatOllama

from schema import Document, ModelOptions
from settings import config

logger = logging.getLogger(__name__)


@partial_transformer
def request_simplfied_text_from_chat_model(
    input: tuple[Document, ModelOptions], prompt_file: str
) -> tuple[Document, str]:
    logger.info("Requesting API for simplified text")

    document, model = input

    llm_chain = _read_prompt_file(prompt_file) | _llm_for_model_name(model)

    response = llm_chain.invoke({"text": document.text})

    return document.model_copy(update={"text": response.text()}), model


@lru_cache
def _read_prompt_file(prompt_file: str) -> PromptTemplate:
    with open(prompt_file, "r") as f:
        return PromptTemplate.from_template("".join(f.readlines()))


def _llm_for_model_name(model: ModelOptions) -> BaseChatModel:
    match model:
        case "gemini-2.5-flash-preview-04-17":
            return ChatGoogleGenerativeAI(
                model=model, temperature=0, max_tokens=None, timeout=None, max_retries=1
            )
        case "cow/gemma2_tools:2b":
            headers = {
                "Authorization": f"Bearer {config['llm_api_key']}",
            }
            return ChatOllama(
                model=model,
                temperature=0,
                disable_streaming=True,
                base_url=config["llm_url"],
                client_kwargs={"headers": headers, "timeout": 360},
            )
        case _:
            raise ValueError(f"Can't find implementation for model {model}")
