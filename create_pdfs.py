import subprocess
from glob import glob
from pathlib import Path

base_directory = Path("./result/text-simplification/generated-simplified")

pandoc_params = [
"-Vgeometry:margin=1in",
"--pdf-engine=lualatex",
'-V', "mainfont=Times New Roman"
]

def to_pdf(mardown: Path):
    as_pdf = mardown.with_suffix(".pdf")
    cmd= ["pandoc",str(mardown),"-o",str(as_pdf)] + pandoc_params

    print(f"Running command: {' '.join(cmd)}")

    subprocess.run(cmd,cwd=base_directory)

if __name__ == "__main__":
    files = glob(base_directory.as_posix() +"/*/*.md")
    as_paths = [Path(file).absolute() for file in files]

    for path in as_paths:
        to_pdf(path)
    