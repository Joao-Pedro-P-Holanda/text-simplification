# default configs
import logging.config
import os
from dotenv import load_dotenv
from pydantic import HttpUrl, SecretStr
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
    "loggers": {"httpx": {"handlers": ["console"], "level": "DEBUG"}},
}

logging.config.dictConfig(logging_config)

_ = load_dotenv()


config: Config = {
    "llm_api_key": SecretStr(os.environ["LLM_API_KEY"]),
    "llm_url": os.environ["OLLAMA_HOST"],
}
