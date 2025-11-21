import os
from pathlib import Path
from dotenv import load_dotenv


_ = load_dotenv()

test_dir = os.path.dirname(__file__)

parent_path = os.path.dirname(test_dir)


def base_original_path():
    return Path(f"{parent_path}/result/data")


def simplified_path(model_name: str):
    return Path(
        f"{parent_path}/result/text-simplification/generated-simplified/{model_name}"
    )
