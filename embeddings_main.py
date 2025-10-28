import os
from csv import DictWriter
from pathlib import Path

import bert_score
from pydantic import BaseModel, computed_field
from glob import glob
from langchain_text_splitters import SentenceTransformersTokenTextSplitter


class Document(BaseModel):
    path: Path
    text: str

    @computed_field
    @property
    def name(self) -> str:
        return self.path.stem


def read_markdown_file(path: str) -> Document:
    with open(path, "r", encoding="utf-8") as file:
        text = "".join(file.readlines())

    return Document(path=Path(path), text=text)


if __name__ == "__main__":
    original_files = sorted(
        [
            Path(x)
            for x in glob("./result-stripped/data/complete/2025_*_stripped.md")
            + glob("./result-stripped/data/simplified/2025_*_stripped.md")
        ],
        key=lambda x: x.stem,
    )

    models = [
        "cow-gemma2_tools-2b",
        "phi4-latest",
        "phi3-latest",
        "llama3.2-latest",
        "gemma3-4b",
        "qwen2.5-14b",
        "deepseek-r1-14b",
        "granite3-dense-2b",
        "granite3-dense-8b",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro-preview-05-06",
    ]

    splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0,
        # Modelo 
        model_name="google-bert/bert-base-multilingual-cased",
        tokens_per_chunk=510,  # descontando 2 de 512 para os tokens especiais CLS e SEP
    )

    with open("embeddings.csv", "a", newline="") as csvfile:
        fieldnames = ["name", "model", "precision", "recall", "f1_score"]
        writer = DictWriter(csvfile, fieldnames=fieldnames)
        # checando se o arquivo não é vazio, não funciona em arquivos xlsx
        if os.stat("embeddings.csv").st_size == 0:
            writer.writeheader()

        for file in original_files:
            for model in models:
                files_for_model = sorted(
                    [
                        Path(x)
                        for x in glob(
                            f"./result-stripped/text-simplification/generated-simplified/{model}/2025_*_stripped.md"
                        )
                    ],
                    key=lambda x: x.stem,
                )
                pairs = list(zip(original_files, files_for_model))

                for pair in pairs[:1]:
                    # Quebrando os dois textos utilizando o Tokenizer
                    original_file = read_markdown_file(pair[0])
                    generated_file = read_markdown_file(pair[1])

                    original_chunks = splitter.split_text(original_file.text)
                    generated_chunks = splitter.split_text(generated_file.text)

                    smaller_chunk_size = min(
                        len(original_chunks), len(generated_chunks)
                    )

                    print(f"Calculando BERTScore para documento {original_file.name} com a versão do modelo {model}")

                    scores = bert_score.score(
                        original_chunks[:smaller_chunk_size],
                        generated_chunks[:smaller_chunk_size],
                        lang="pt",
                    )

                    writer.writerow(
                        {
                            "name": original_file.name,
                            "model": model,
                            "precision": scores[0].mean().item(),
                            "recall": scores[1].mean().item(),
                            "f1_score": scores[2].mean().item(),
                        }
                    )