# default configs
from functools import lru_cache
import logging.config
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from schema import Config

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": "DEBUG",
            "filename": ".log",
            "mode": "a",
        },
    },
    "root": {
        "handlers": [
            "console",
            "file",
        ],
        "level": "DEBUG",
    },
}

logging.config.dictConfig(logging_config)

_ = load_dotenv()


@lru_cache
def read_prompt_file(path: str):
    with open(path) as file:
        return "".join(file.readlines())


prompt = read_prompt_file("./prompt_simplify_document.txt")


config: Config = {
    "models": {"cow/gemma2_tools:2b"},
    "llm_api_key": SecretStr(os.environ["LLM_API_KEY"]),
    "gemini_api_key": SecretStr(os.environ["GEMINI_API_KEY"]),
    "gemini_base_url": "https://generativelanguage.googleapis.com/v1beta/models/",
    "llm_url": "http://ollama.atlab.ufc.br:8080/ollama/api/generate",
    "notices_url": "https://parquetecnologico.ufc.br/pt/edital/",
}
