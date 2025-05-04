"""
This is the entrypoint for the textual simplification performed by LLMS
"""

from pipelines import (
    simplify_pdf_file_with_api_model,
    simplify_pdf_file_with_self_hosted_model,
)


if __name__ == "__main__":
    print(
        simplify_pdf_file_with_self_hosted_model("complete/edital-aCAo-de-extensAo.pdf")
    )
    print(simplify_pdf_file_with_api_model("complete/edital-aCAo-de-extensAo.pdf"))
