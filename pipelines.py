from pathlib import Path
import os

from settings import config
from file_processing_steps import (
    convert_markdown_text_to_markdown_file,
    convert_pdf_file_to_markdown_text,
)
from request_steps import create_prompt_from_target_text, request_simplfied_text

DATA_DIR = Path(os.path.join(os.path.dirname(__file__), "data"))


simplify_pdf_file = (
    convert_pdf_file_to_markdown_text
    >> create_prompt_from_target_text
    >> request_simplfied_text(
        url=config["llm_url"],
        model="cow/gemma2_tools:2b",
        token=config["llm_api_key"].get_secret_value(),
    )
    >> convert_markdown_text_to_markdown_file("./result/converted.md")
)
