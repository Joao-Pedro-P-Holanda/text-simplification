from contextvars import ContextVar
import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid1

from gloe import Transformer, condition
from gloe.utils import forward
from gloe.collection import Map

from file_processing_steps import (
    save_document_text_on_markdown_file,
    raise_non_markdown_error,
    read_markdown_file,
    store_results_as_csv,
    remove_think_tags,
)
from metrics_steps import (
    extract_document_statistics_port_parser,
    list_complex_words,
    transform_document_to_metric_operations,
    extract_min_nilc_metrix_ud,
)
from request_steps import (
    request_simplified_text_from_chat_model,
    generate_chunks_for_text,
)
from schema import ModelOptions
from utils import set_execution_uuid, zip_to_one, pick_first, pick_second

DATA_DIR = Path(os.path.join(os.path.dirname(__file__), "data"))


@condition
def is_markdown(path: str) -> bool:
    return path.endswith(".md")


uuid_context: ContextVar[UUID] = ContextVar("uuid_context")

simplify_pdf_files_with_model: Transformer[
    tuple[list[str], ModelOptions], list[Any]
] = (
    set_execution_uuid(uuid_context=uuid_context)
    >> zip_to_one
    >> (
        Map(
            forward[tuple[str, ModelOptions]]()
            >> (
                pick_first
                >> is_markdown.Then(read_markdown_file).Else(raise_non_markdown_error),
                pick_second,
            )
        )
    )
    >> Map(
        generate_chunks_for_text
        >> request_simplified_text_from_chat_model(
            prompt_file="prompt_simplify_document.txt",
        )
        >> remove_think_tags
        >> save_document_text_on_markdown_file(
            doc_type="generated-simplified", execution_uuid=uuid_context
        )
    )
)

extract_udpipe_nilc_metrix_from_original_complete_texts: Transformer[
    list[str], None
] = Map(read_markdown_file >> extract_min_nilc_metrix_ud) >> store_results_as_csv(
    task_type="ud-pipe", doc_type="reference-complete"
)

extract_udpipe_nilc_metrix_from_original_simplified_texts: Transformer[
    list[str], None
] = Map(read_markdown_file >> extract_min_nilc_metrix_ud) >> store_results_as_csv(
    task_type="ud-pipe", doc_type="reference-simplified"
)

extract_udpipe_nilc_metrix_from_generated_simplified_texts: Transformer[
    list[str], None
] = Map(read_markdown_file >> extract_min_nilc_metrix_ud) >> store_results_as_csv(
    task_type="ud-pipe", doc_type="generated-simplified"
)

extract_metrics_from_generated_texts_port_tokenizer: Transformer[list[str], None] = Map(
    read_markdown_file
    >> transform_document_to_metric_operations
    >> list_complex_words(frequencies_file="./data/frequencias_todos_os_corpora.pkl")
    >> extract_document_statistics_port_parser
) >> store_results_as_csv(
    task_type="readability-indexes", doc_type="generated-simplified", mode="w"
)

extract_metrics_from_complete_texts_port_tokenizer: Transformer[list[str], None] = Map(
    read_markdown_file
    >> transform_document_to_metric_operations
    >> list_complex_words(frequencies_file="./data/frequencias_todos_os_corpora.pkl")
    >> extract_document_statistics_port_parser
) >> store_results_as_csv(
    task_type="readability-indexes", doc_type="reference-complete", mode="w"
)

extract_metrics_from_already_simplified_texts_port_tokenizer: Transformer[
    list[str], None
] = Map(
    read_markdown_file
    >> transform_document_to_metric_operations
    >> list_complex_words(frequencies_file="./data/frequencias_todos_os_corpora.pkl")
    >> extract_document_statistics_port_parser
) >> store_results_as_csv(
    task_type="readability-indexes", doc_type="reference-simplified", mode="w"
)
