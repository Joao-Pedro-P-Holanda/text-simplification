from pathlib import Path
import time
import httpx
from glob import glob


if __name__ == "__main__":
    models = glob("./result/text-simplification/generated-simplified/gemma3-4b")

    params = {
        "model": "portuguese-porttinari-ud-2.15-241121",
        "tokenizer": "presegmented-input",
        "tagger": "",
        "parser": "",
    }

    with httpx.Client(params=params, timeout=None) as client:
        for model in models:
            files = glob(model + "/*_stripped.md")
            for file in files:
                print(f"Requesting file {file}")
                output_path = Path(file).with_suffix(".conllu")
                response = client.post(
                    "https://lindat.mff.cuni.cz/services/udpipe/api/process",
                    files={"data": open(file, "rb")},
                )

                response.raise_for_status()

                with open(output_path, "w") as f:
                    print(f"Writing result at {output_path}")
                    f.write(response.json()["result"])

            time.sleep(10)
