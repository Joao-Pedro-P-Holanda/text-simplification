import logging
from gloe import partial_transformer, transformer
import pathlib
import pymupdf4llm
import os

logger = logging.getLogger(__name__)


@transformer
def process_pdf_images(): ...


@transformer
def process_pdf_tables(): ...


@transformer
def convert_pdf_file_to_markdown_text(path: str) -> str:
    logger.info(f"Converting pdf file at {path} to markdown text")
    text = pymupdf4llm.to_markdown(f"data/{path}")

    return text


@transformer
def convert_pdf_file_to_markdown_file(path: str) -> pathlib.Path:
    text = pymupdf4llm.to_markdown(f"data/{path}")

    output = pathlib.Path("./result") / pathlib.Path(path).name
    output.write_bytes(text.encode())

    return output


@partial_transformer
def convert_markdown_text_to_markdown_file(content: str, output_path: str):
    os.makedirs("./result", exist_ok=True)
    with open(output_path, "w") as file:
        file.write(content)
