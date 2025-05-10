import csv
import logging
from typing import Iterable
from gloe import partial_transformer, transformer
import pathlib
import pymupdf4llm
import os

from schema import Document, DocumentResultModel, DocumentType

logger = logging.getLogger(__name__)


@partial_transformer
def store_results_as_csv(
    results: Iterable[DocumentResultModel], doc_type: DocumentType
):
    store_path = f"./result/{doc_type}.csv"

    with open(store_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=None)  # type:ignore

        for i, result in enumerate(results):
            data = result.model_dump()
            if i == 0:
                writer.fieldnames = data.keys()

                writer.writeheader()

            writer.writerow(data)


@transformer
def process_pdf_images(): ...


@transformer
def process_pdf_tables(): ...


@transformer
def read_markdown_file(path: str) -> Document:
    with open(path, "r") as file:
        text = "".join(file.readlines())

    return Document(path=path, text=text)


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
