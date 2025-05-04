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
def request_simplfied_text_from_self_hosted(
    prompt: str, url: str, model: ModelOptions, token: str
) -> str:
    logger.info("Requesting API for simplified text")
    if model == "cow/gemma2_tools:2b":
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        data = {"model": model, "prompt": prompt, "stream": False}

        response = httpx.post(url, headers=headers, json=data, timeout=60).json()
        return response["response"]
    elif model == "gemini-2.5-flash-preview-04-17":
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        params = {"key": token}

        data = {"contents": [{"parts": [{"text": prompt}]}]}

        response = httpx.post(
            "".join([url, f"{model}:generateContent"]),
            params=params,
            headers=headers,
            json=data,
            timeout=60,
        )
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
