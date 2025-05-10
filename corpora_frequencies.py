"""
Select the top words in the linguateca word frequency file, using only alphabetic
characters
"""

import logging
import pickle
import httpx
from utils import is_valid_word
import settings

_ = settings

MAX_WORDS_SELECTED = 5000

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Requesting word frequencies text file")
    with httpx.stream(
        "GET",
        "https://www.linguateca.pt/acesso/tokens/formas.todos.txt",
        verify="./linguateca-pt-chain.pem",
    ) as response:
        frequencies: dict[str, int] = {}
        logger.info("Storing top frequencies")
        for line in response.iter_lines():
            splitted = line.split("\t")
            if is_valid_word(splitted[1]):
                frequencies[splitted[1]] = int(splitted[0])
            if len(frequencies) == MAX_WORDS_SELECTED:
                break

        with open("./data/frequencias_todos_os_corpora.pkl", "wb") as file:
            pickle.dump(frequencies, file)

        logger.info(f"stored {MAX_WORDS_SELECTED} words")
