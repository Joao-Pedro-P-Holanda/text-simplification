import logging
from functools import lru_cache

from gloe import partial_transformer, transformer
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware.types import after_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from schema import Document, ModelOptions
from settings import config

logger = logging.getLogger(__name__)


@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    messages_to_remove = 4
    if len(messages) > messages_to_remove:
        # mantendo uma das mensagens para manter o que já foi realizado no contexto
        return {
            "messages": [
                RemoveMessage(id=m.id) for m in messages[: messages_to_remove - 1]
            ]
        }
    return None


@partial_transformer
def request_simplified_text_from_chat_model(
    input: tuple[Document, ModelOptions], prompt_file: str
) -> tuple[Document, str]:
    document, model, *_ = input

    llm_chain = _llm_for_model_name(model)

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    logger.info(f"Requesting model {model} for simplified file {document.name}")

    llm_chain.invoke(
        {"messages": [SystemMessage(_read_prompt_file(prompt_file))]}, config
    )

    chunks = document.langchain_documents

    response = ""

    for chunk in chunks:
        part = llm_chain.invoke({"messages": [HumanMessage(chunk)]}, config)[
            "messages"
        ][-1].content
        response += part + "\n"

    return document.model_copy(update={"text": response}), model


@transformer
def generate_chunks_for_text(
    input: tuple[Document, ModelOptions],
) -> tuple[Document, str]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "header 2")], strip_headers=False
    )

    input[0].langchain_documents = [
        doc.page_content for doc in splitter.split_text(input[0].text)
    ]

    # As transformações seguintes são feitas para manter cada chunk com, aproximadamente,
    # 2000 tokens. Considerando a aproximação de 4 letras por token
    # https://platform.openai.com/tokenizer. Assim, as 4 mensagens + prompt de sistema
    # vão ocupar aproximadamente metade da menor janela de contexto (16384)

    # combinando seções pequenas
    input[0].langchain_documents = _merge_small_chunks(input[0].langchain_documents)

    # Quebrando seções muito grandes com o RecursiveCharacterTextSplitter,
    input[0].langchain_documents = _break_large_sections(input[0].langchain_documents)

    return (input[0], input[1])


def _merge_small_chunks(chunks: list[str], threshold: int = 8000) -> list[str]:
    if not chunks:
        return []

    new_chunks = []

    current_chunks = []

    for chunk in chunks:
        if sum([len(merged) for merged in current_chunks]) < threshold:
            current_chunks.append(chunk)
        else:
            new_chunks.append("\n".join(current_chunks))
            current_chunks = [chunk]

    if current_chunks:
        new_chunks.append("\n".join(current_chunks))

    return new_chunks


def _break_large_sections(chunks: list[str], threshold: int = 8000) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=threshold, chunk_overlap=0)

    new_chunks: list[str] = []

    for chunk in chunks:
        new_chunks.extend([text for text in splitter.split_text(chunk)])

    return new_chunks


@lru_cache
def _read_prompt_file(prompt_file: str) -> str:
    with open(prompt_file, "r") as f:
        return "".join(f.readlines())


def _llm_for_model_name(model: ModelOptions) -> BaseChatModel:
    temperature = 1
    match model:
        case "gemini-2.5-flash-preview-04-17" | "gemini-2.5-pro-preview-05-06":
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                timeout=None,
                max_retries=1,
            )
        case (
            # open web ui
            "cow/gemma2_tools:2b"
            | "phi4:latest"
            | "phi3:latest"
            | "llama3.2:latest"
            | "gemma3:4b"
            | "qwen2.5:14b"
            | "deepseek-r1:14b"
            | "granite3-dense:2b"
            | "granite3-dense:8b"
            # open router
            | "google/gemma-3n-e4b-it"
            | "microsoft/phi-4"
            | "meta-llama/llama-3.2-3b-instruct"
            | "qwen/qwen3-14b"
            | "ibm-granite/granite-4.0-h-micro"
            | "deepseek/deepseek-v3.2"
            | "mistralai/ministral-8b"
            | "google/gemini-2.5-flash"
            | "google/gemini-2.5-pro"
        ):
            langchain_model = ChatOpenAI(
                model=model,
                temperature=temperature,
                base_url=config["llm_url"],
                timeout=None,
                api_key=config["llm_api_key"],
            )
            agent = create_agent(
                langchain_model,
                middleware=[delete_old_messages],
                checkpointer=InMemorySaver(),
            )

            return agent
        case _:
            raise ValueError(f"Can't find implementation for model {model}")
