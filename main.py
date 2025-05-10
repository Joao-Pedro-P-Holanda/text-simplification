"""
This is the entrypoint for the textual simplification performed by LLMS
"""

from collections import namedtuple

from pipelines import (
    extract_metrics_from_saved_texts,
)

DocPaths = namedtuple("DocPaths", ["original", "reference", "simplified"])

documents_path: list[tuple[str, str, str]] = []

if __name__ == "__main__":
    # TODO: Use gloe map to perform pipeline in all files

    # simplify_pdf_file_with_api_model("complete/edital-spin-offs-partec-ufc.pdf")
    extract_metrics_from_saved_texts(["./result/2_converted_gemini.md"])
