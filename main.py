"""
This is the entrypoint for the textual simplification performed by LLMS
"""

from collections import namedtuple
from glob import glob

from gloe.collection import Map

from file_processing_steps import convert_pdf_file_to_markdown_file
from pipelines import (
    compute_embeddings_similarity_for_complete_and_generated_texts,
    extract_metrics_from_complete_texts,
    extract_metrics_from_generated_texts,
    simplify_pdf_files_with_model,
)

DocPaths = namedtuple("DocPaths", ["original", "reference", "simplified"])


if __name__ == "__main__":
    complete_documents = glob("./result/data/complete/2025_*.md")

    generated_simplified_models = glob(
        "./result/text-simplification/generated-simplified/*"
    )

    for generated_simplified in generated_simplified_models:
        files = glob(f"{generated_simplified}/2025_*.md")
        pairs = list(zip(sorted(complete_documents), sorted(files)))
        compute_embeddings_similarity_for_complete_and_generated_texts(pairs)

    # simplify_pdf_files_with_model(
    #     (complete_documents, "gemini-2.5-flash-preview-04-17")
    # )

    # TODO: first convert all pdfs to markdown, then generate the simplified version for all complete documents, then compute the embeddings for the complete, original simplified and generated simplified versions between them and store the results marking which model generated what, then compute the Readability scores metrics marking which model interpreted the texts and store the results,
    # TODO: then compute the D-SARI metrics between the documents that have a reference and store the results
    # TODO: then compute the NILC-Metrix for the three versions and store the results, marking the model used

    # extract_metrics_from_complete_texts(
    #     [
    #         "./result/text-simplification/generated-simplified/gemini-2.5-flash-preview-04-17/4_ufc_inova_2025.md"
    #     ]
    # )
