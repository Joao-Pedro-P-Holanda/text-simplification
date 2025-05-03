import httpx
import logging
from gloe import partial_transformer, transformer
from settings import prompt
from schema import ModelOptions

logger = logging.getLogger(__name__)


@transformer
def create_prompt_from_target_text(text: str) -> str:
    logger.info("Inserting target text on prompt")
    return prompt.replace("[Text goes here]", text)


@partial_transformer
def request_simplfied_text(
    prompt: str, url: str, model: ModelOptions, token: str
) -> str:
    logger.info("Requesting API for simplified text")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data = {"model": model, "prompt": prompt, "stream": False}

    response = httpx.post(url, headers=headers, json=data, timeout=60).json()
    return response["response"]
