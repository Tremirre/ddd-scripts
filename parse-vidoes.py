import json
import sys
import pathlib
import gdown
import subprocess

import patoolib


tmp_dir = pathlib.Path("~/AppData/Local/Temp/ddd/out/").expanduser()
URL_TEMPLATE = "https://drive.google.com/uc?id={}"

if __name__ == "__main__":
    # tmp_dir.mkdir(parents=True, exist_ok=True)
    # for file in tmp_dir.glob("*"):
    #     file.unlink()
    ids_path = sys.argv[1]
    with open(ids_path, "r") as f:
        ids = json.load(f)
    print(f"Loaded {len(ids)} ids from {ids_path}")
    for file_id in ids:
        url = URL_TEMPLATE.format(file_id)
        output_file = tmp_dir / f"{file_id}.7z"
        if not output_file.exists():
            gdown.download(url, output=str(tmp_dir / f"{file_id}.7z"), quiet=False)

        print(f"Extracting {file_id} to {tmp_dir}")
        patoolib.extract_archive(
            str(output_file), outdir=str(tmp_dir), verbosity=1, interactive=False
        )
        output_file.unlink()
        extracted_file = list(tmp_dir.glob("*.hdf5"))[0]

        res = subprocess.run(
            [
                "python",
                "exporter.py",
                "--input",
                str(extracted_file),
                "--output",
                str(tmp_dir / f"{file_id}.npz"),
            ],
            check=False,
            capture_output=True,
        )
        if res.returncode != 0:
            print(f"Failed to export {file_id}: {res.stderr!r}")
            exit(1)

        print(f"Exported {file_id} to {tmp_dir / f'{file_id}.npz'}")
        extracted_file.unlink()

        subprocess.run(
            [
                "python",
                "converter.py",
                "--input",
                str(tmp_dir / f"{file_id}.npz"),
                "--output",
                f"videos/{file_id}.mp4",
            ],
            check=True,
            capture_output=True,
        )
        print(f"Converted {file_id} to videos/{file_id}.mp4")
        (tmp_dir / f"{file_id}.npz").unlink()

        exit(0)

    print(tmp_dir.resolve())
    print("Hello World")
