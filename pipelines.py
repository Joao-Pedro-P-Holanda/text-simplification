from pathlib import Path
import os

from settings import config
from file_processing_steps import (
    convert_markdown_text_to_markdown_file,
    convert_pdf_file_to_markdown_text,
)
from request_steps import (
    create_prompt_from_target_text,
    request_simplfied_text_from_self_hosted,
)

DATA_DIR = Path(os.path.join(os.path.dirname(__file__), "data"))


simplify_pdf_file_with_self_hosted_model = (
    convert_pdf_file_to_markdown_text
    >> create_prompt_from_target_text
    >> request_simplfied_text_from_self_hosted(
        url=config["llm_url"],
        model="cow/gemma2_tools:2b",
        token=config["llm_api_key"].get_secret_value(),
    )
    >> convert_markdown_text_to_markdown_file("./result/converted.md")
)

simplify_pdf_file_with_api_model = (
    convert_pdf_file_to_markdown_text
    >> create_prompt_from_target_text
    >> request_simplfied_text_from_self_hosted(
        url=config["gemini_base_url"],
        model="gemini-2.5-flash-preview-04-17",
        token=config["gemini_api_key"].get_secret_value(),
    )
    >> convert_markdown_text_to_markdown_file("./result/converted_gemini.md")
)
