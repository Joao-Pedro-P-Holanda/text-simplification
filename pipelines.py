import os
from pathlib import Path
from typing import Any

from gloe import Transformer, condition
from gloe.utils import forward
from gloe.collection import Map

from embeddings_steps import compare_embedded_sentences_similarity
from file_processing_steps import (
    save_document_text_on_markdown_file,
    convert_pdf_file_to_markdown_text,
    read_markdown_file,
    store_results_as_csv,
)
from metrics_steps import (
    extract_document_statistics,
    extract_document_statistics_port_parser,
    group_documents_by_model,
    list_complex_words,
    extract_min_nilc_metrix,
    transform_document_to_metric_operations,
)
from request_steps import (
    request_simplfied_text_from_chat_model,
    generate_documents_for_texts,
)
from schema import ModelOptions
from settings import load_spacy_model
from utils import convert_to_list, zip_to_one, pick_first, pick_second

DATA_DIR = Path(os.path.join(os.path.dirname(__file__), "data"))


@condition
def is_markdown(path: str) -> bool:
    return path.endswith(".md")


simplify_pdf_files_with_model: Transformer[
    tuple[list[str], ModelOptions], list[Any]
] = (
    zip_to_one
    >> (
        Map(
            forward[tuple[str, ModelOptions]]()
            >> (
                pick_first
                >> is_markdown.Then(read_markdown_file).Else(
                    convert_pdf_file_to_markdown_text
                ),
                pick_second,
            )
        )
    )
    >> generate_documents_for_texts
    >> Map(
        request_simplfied_text_from_chat_model(
            prompt_file="prompt_simplify_document.txt",
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

extract_nilc_metrix_from_original_complete_texts: Transformer[list[str], None] = Map(
    read_markdown_file
    >> transform_document_to_metric_operations
    >> extract_min_nilc_metrix
) >> store_results_as_csv(task_type="nilc-metrix", doc_type="reference-complete")

# perform extraction batches per name
extract_nilc_metrix_from_generated_texts: Transformer[list[str], list[None]] = (
    Map(read_markdown_file >> transform_document_to_metric_operations)
    >> group_documents_by_model
    >> Map(
        Map(
            extract_min_nilc_metrix
            >> convert_to_list
            >> store_results_as_csv(
                task_type="nilc-metrix", doc_type="generated-simplified", mode="a"
            )
        )
    )
)


extract_nilc_metrix_from_original_simplified_texts: Transformer[list[str], None] = Map(
    read_markdown_file
    >> transform_document_to_metric_operations
    >> extract_min_nilc_metrix
) >> store_results_as_csv(task_type="nilc-metrix", doc_type="reference-simplified")


extract_metrics_from_generated_texts_port_tokenizer: Transformer[list[str], None] = Map(
    read_markdown_file
    >> transform_document_to_metric_operations
    >> list_complex_words(frequencies_file="./data/frequencias_todos_os_corpora.pkl")
    >> extract_document_statistics_port_parser
) >> store_results_as_csv(
    task_type="readability-indexes", doc_type="generated-simplified", mode="a"
)


extract_metrics_from_generated_texts: Transformer[list[str], None] = Map(
    read_markdown_file
    >> transform_document_to_metric_operations
    >> list_complex_words(frequencies_file="./data/frequencias_todos_os_corpora.pkl")
    >> extract_document_statistics(
        nlp_loader=load_spacy_model, model_name="pt_core_news_lg"
    )
) >> store_results_as_csv(
    task_type="readability-indexes", doc_type="generated-simplified"
)

extract_metrics_from_complete_texts_port_tokenizer: Transformer[list[str], None] = Map(
    read_markdown_file
    >> transform_document_to_metric_operations
    >> list_complex_words(frequencies_file="./data/frequencias_todos_os_corpora.pkl")
    >> extract_document_statistics_port_parser
) >> store_results_as_csv(
    task_type="readability-indexes", doc_type="reference-complete", mode="a"
)


extract_metrics_from_complete_texts: Transformer[list[str], None] = Map(
    read_markdown_file
    >> transform_document_to_metric_operations
    >> list_complex_words(frequencies_file="./data/frequencias_todos_os_corpora.pkl")
    >> extract_document_statistics(
        nlp_loader=load_spacy_model, model_name="pt_core_news_lg"
    )
) >> store_results_as_csv(
    task_type="readability-indexes", doc_type="reference-complete"
)


extract_metrics_from_already_simplified_texts_port_tokenizer: Transformer[
    list[str], None
] = Map(
    read_markdown_file
    >> transform_document_to_metric_operations
    >> list_complex_words(frequencies_file="./data/frequencias_todos_os_corpora.pkl")
    >> extract_document_statistics_port_parser
) >> store_results_as_csv(
    task_type="readability-indexes", doc_type="reference-simplified", mode="a"
)

extract_metrics_from_already_simplfied_texts: Transformer[list[str], None] = Map(
    read_markdown_file
    >> transform_document_to_metric_operations
    >> list_complex_words(frequencies_file="./data/frequencias_todos_os_corpora.pkl")
    >> extract_document_statistics(
        nlp_loader=load_spacy_model, model_name="pt_core_news_lg"
    )
) >> store_results_as_csv(
    task_type="readability-indexes", doc_type="reference-simplified"
)
