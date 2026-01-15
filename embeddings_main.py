import os
from csv import DictWriter
from pathlib import Path

import bert_score
from dotenv import load_dotenv
from glob import glob
from langchain_text_splitters import SentenceTransformersTokenTextSplitter


def read_stripped_markdown_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as file:
        text = "".join(file.readlines())

    return text


if __name__ == "__main__":
    load_dotenv()
    ORIGINAL_DATA_DIR = os.getenv("DATA_BASE_DIR","data")
    RESULT_DIR = os.getenv("RESULTS_BASE_DIR",'result')

    original_files = {
            Path(x).name.replace("_simplificado_stripped",""): Path(x)
            for x in glob(f"./{ORIGINAL_DATA_DIR}/**/*simplificado_stripped.md",recursive=True)
        }

    generated_files_with_model_name = [(Path(x).parent.name ,Path(x)) for x in glob(f"./{RESULT_DIR}/**/*_stripped.md",recursive=True)]

    splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0,
        # Modelo 
        model_name="google-bert/bert-base-multilingual-cased",
        tokens_per_chunk=510,  # descontando 2 de 512 para os tokens especiais CLS e SEP
    )

    with open("embeddings.csv", "w", newline="") as csvfile:
        fieldnames = ["name", "model", "precision", "recall", "bert_score"]
        writer = DictWriter(csvfile, fieldnames=fieldnames)
        # checando se o arquivo não é vazio, não funciona em arquivos xlsx
        if os.stat("embeddings.csv").st_size == 0:
            writer.writeheader()

        for model,file in generated_files_with_model_name:
            matching_original = original_files.get(file.name.replace("_stripped",""))

            if not matching_original:
                print(f"File {file.name} don't have a corresponding simplification, skipping")
                continue

            # Quebrando os dois textos utilizando o Tokenizer
            original_simplification = read_stripped_markdown_file(matching_original)
            generated_simplification = read_stripped_markdown_file(file)

            original_chunks = splitter.split_text(original_simplification)
            generated_chunks = splitter.split_text(generated_simplification)

            smaller_chunk_size = min(
                len(original_chunks), len(generated_chunks)
            )

            print(f"Calculando BERTScore para documento {file.name} com a versão do modelo {model}")

            scores = bert_score.score(
                original_chunks[:smaller_chunk_size],
                generated_chunks[:smaller_chunk_size],
                lang="pt",
            )

            writer.writerow(
                {
                    "name": file.name,
                    "model": model,
                    "precision": scores[0].mean().item(),
                    "recall": scores[1].mean().item(),
                    "bert_score": scores[2].mean().item(),
                }
            )