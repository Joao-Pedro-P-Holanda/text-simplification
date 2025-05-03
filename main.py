"""
This is the entrypoint for the textual simplification performed by LLMS
"""

from pipelines import simplify_pdf_file


if __name__ == "__main__":
    print(simplify_pdf_file("complete/edital-aCAo-de-extensAo.pdf"))
