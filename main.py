"""
This is the entrypoint for the textual simplification performed by LLMS
"""

from pipelines import (
    extract_metrics_from_saved_text,
)


if __name__ == "__main__":
    print(extract_metrics_from_saved_text("./result/converted_gemini.md"))
