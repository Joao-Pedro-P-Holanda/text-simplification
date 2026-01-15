import csv
import logging
import os
import pathlib
from collections.abc import Iterable
from typing import Literal

import pymupdf4llm
from contextvars import ContextVar
from gloe import partial_transformer, transformer
from uuid import UUID

from schema import Document, DocumentResultModel, DocumentType, TaskType, ModelOptions
import re


logger = logging.getLogger(__name__)


@partial_transformer
def store_results_as_csv(
    results: Iterable[DocumentResultModel],
    task_type: TaskType,
    doc_type: DocumentType | None = None,
    mode: Literal["a", "w"] = "w",
):
    store_path = pathlib.Path(
        f"./result/{task_type}{'/' + doc_type if doc_type else '/result'}.csv"
    )
    os.makedirs(store_path.parent, exist_ok=True)
    number_of_lines = 0
    if os.path.exists(store_path):
        with open(store_path, "r") as f:
            number_of_lines = len(f.readlines())

    with open(store_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=None)  # type:ignore

        writer.fieldnames = results[0].model_dump().keys()
        if mode == "w" or (mode == "a" and number_of_lines == 0):
            writer.writeheader()
            number_of_lines += 1

        for result in results:
            data = result.model_dump()
            writer.writerow(data)


@transformer
def read_markdown_file(path: str) -> Document:
    logger.info(f"Reading file {path}")
    with open(path, "r", encoding="utf-8") as file:
        text = "".join(file.readlines())

    return Document(path=path, text=text)


@transformer
def raise_non_markdown_error(path: str) -> Document:
    raise ValueError(f"Sended non markdown file {path}")


@transformer
def convert_pdf_file_to_markdown_file(path: str) -> pathlib.Path:
    text = pymupdf4llm.to_markdown(path)
    as_sys_path = pathlib.Path("./result") / pathlib.Path(path)

    as_sys_path.parent.mkdir(parents=True, exist_ok=True)

    output = as_sys_path.with_suffix(".md")
    output.write_bytes(text.encode())

    return output


@partial_transformer
def save_document_text_on_markdown_file(
    input: tuple[Document, str], doc_type: DocumentType, execution_uuid: ContextVar[UUID]
) -> None:
    document, model = input

    # replaces double dots to save correctly on Windows
    model_path = re.sub("/|:", "-", model)
    store_directory = (
        f"./result/text-simplification/{doc_type}/{execution_uuid.get()}/{model_path}"
    )

    os.makedirs(
        store_directory,
        exist_ok=True,
    )
    with open(
        f"{store_directory}/{document.name}.md",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(document.text)


@transformer
def save_file_without_formatting(document: Document) -> None:
    original_path = pathlib.Path(document.path)
    new_path = original_path.with_stem(original_path.stem + "_stripped")
    logger.info(f"Stripping markdown formating for file {document.path}")
    with open(new_path, "w") as stripped_file:
        stripped_file.write(document.text)


@transformer
def remove_think_tags(
    input: tuple[Document, ModelOptions],
) -> tuple[Document, ModelOptions]:
    document, model = input
    if model == "deepseek-r1:14b":
        document.text = re.sub(
            re.compile("<think>.*?</think>", re.DOTALL), "", document.text
        )

    return document, model
