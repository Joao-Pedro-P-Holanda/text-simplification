# default configs
from functools import lru_cache
import logging.config
import os
import spacy
from dotenv import load_dotenv
from pydantic import SecretStr
from schema import Config
import spacy_syllables

_ = spacy_syllables

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
    "loggers": {"httpx": {"handlers": ["console"], "level": "DEBUG"}},
}

logging.config.dictConfig(logging_config)

_ = load_dotenv()


@lru_cache
def load_spacy_model(name: str):
    result = spacy.load(name)
    result.add_pipe("syllables", after="morphologizer", config={"lang": "pt_BR"})
    return result


config: Config = {
    "models": {"cow/gemma2_tools:2b"},
    "llm_api_key": SecretStr(os.environ["LLM_API_KEY"]),
    "llm_url": "http://ollama.atlab.ufc.br:8080/ollama/",
}
