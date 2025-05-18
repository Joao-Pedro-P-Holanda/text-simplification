import os
from pathlib import Path
from typing import Any

from gloe import Transformer
from gloe.utils import forward
from gloe.collection import Map

from embeddings_steps import compare_embedded_sentences_similarity
from file_processing_steps import (
    save_document_text_on_markdown_file,
    convert_pdf_file_to_markdown_text,
    read_markdown_file,
    store_results_as_csv,
)
from metrics_steps import extract_document_statistics, list_complex_words
from request_steps import (
    request_simplfied_text_from_chat_model,
)
from schema import ModelOptions
from settings import load_spacy_model
from utils import zip_to_one, pick_first, pick_second

DATA_DIR = Path(os.path.join(os.path.dirname(__file__), "data"))

nlp = load_spacy_model("pt_core_news_lg")


simplify_pdf_files_with_model: Transformer[
    tuple[list[str], ModelOptions], list[Any]
] = (
    zip_to_one
    >> (
        Map(
            forward[tuple[str, ModelOptions]]()
            >> (
                pick_first >> convert_pdf_file_to_markdown_text,
                pick_second,
            )
        )
    )
    >> Map(
        request_simplfied_text_from_chat_model(
            prompt_file="prompt_simplify_document.txt"
        )
        >> save_document_text_on_markdown_file(doc_type="generated-simplified")
    )
)


compute_embeddings_similarity_for_complete_and_generated_texts: Transformer[
    list[tuple[str, str]], None
] = Map(
    forward[tuple[str, str]]()
    >> (pick_first >> read_markdown_file, pick_second >> read_markdown_file)
    >> compare_embedded_sentences_similarity("nomic-ai/nomic-embed-text-v2-moe")
) >> store_results_as_csv(
    task_type="embedding-similarity",
)

extract_metrics_from_generated_texts: Transformer[list[str], None] = Map(
    read_markdown_file
    >> list_complex_words(frequencies_file="./data/frequencias_todos_os_corpora.pkl")
    >> extract_document_statistics(nlp=nlp)
) >> store_results_as_csv(
    task_type="readability-indexes", doc_type="generated-simplified"
)

extract_metrics_from_complete_texts: Transformer[list[str], None] = Map(
    read_markdown_file
    >> list_complex_words(frequencies_file="./data/frequencias_todos_os_corpora.pkl")
    >> extract_document_statistics(nlp=nlp)
) >> store_results_as_csv(
    task_type="readability-indexes", doc_type="reference-complete"
)

extract_metrics_from_already_simplfied_texts: Transformer[list[str], None] = Map(
    read_markdown_file
    >> list_complex_words(frequencies_file="./data/frequencias_todos_os_corpora.pkl")
    >> extract_document_statistics(nlp=nlp)
) >> store_results_as_csv(
    task_type="readability-indexes", doc_type="reference-simplified"
)
