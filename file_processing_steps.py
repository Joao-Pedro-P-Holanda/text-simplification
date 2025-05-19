import csv
import logging
import os
import pathlib
from collections.abc import Iterable

import pymupdf4llm
from gloe import partial_transformer, transformer

from schema import Document, DocumentResultModel, DocumentType, TaskType

logger = logging.getLogger(__name__)


@partial_transformer
def store_results_as_csv(
    results: Iterable[DocumentResultModel],
    task_type: TaskType,
    doc_type: DocumentType | None = None,
):
    store_path = pathlib.Path(
        f"./result/{task_type}{'/' + doc_type if doc_type else '/result'}.csv"
    )
    os.makedirs(store_path.parent, exist_ok=True)

    with open(store_path, "w",newline='') as f:
        writer = csv.DictWriter(f, fieldnames=None)  # type:ignore

        for i, result in enumerate(results):
            data = result.model_dump()
            if i == 0:
                writer.fieldnames = data.keys()

                writer.writeheader()

            writer.writerow(data)


@transformer
def read_markdown_file(path: str) -> Document:
    logger.info(f"Reading file {path}")
    with open(path, "r") as file:
        text = "".join(file.readlines())

    return Document(path=path, text=text)


@transformer
def convert_pdf_file_to_markdown_text(path: str) -> Document:
    logger.info(f"Converting pdf file at {path} to markdown text")
    text = pymupdf4llm.to_markdown(path)

    return Document(text=text, path=path)


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
    input: tuple[Document, str], doc_type: DocumentType
) -> None:
    document, model = input

    os.makedirs(f"./result/text-simplification/{doc_type}/{model}", exist_ok=True)
    with open(
        f"./result/text-simplification/{doc_type}/{model}/{document.name}.md", "w"
    ) as file:
        file.write(document.text)
