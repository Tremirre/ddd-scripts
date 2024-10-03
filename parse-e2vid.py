import json
import logging
import pathlib
import subprocess
import sys

import gdown
import patoolib

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TMP_DIR = pathlib.Path("~/AppData/Local/Temp/ddd/out/").expanduser()
URL_TEMPLATE = "https://drive.google.com/uc?id={}"
OUT_FOLDER = pathlib.Path("videos")

if __name__ == "__main__":

    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    name_mapping_file = OUT_FOLDER / "names.json"
    name_mapping = {}
    if name_mapping_file.exists():
        with open(name_mapping_file, "r") as f:
            name_mapping = json.load(f)

    for file in TMP_DIR.glob("*"):
        file.unlink()

    ids_path = sys.argv[1]
    with open(ids_path, "r") as f:
        ids = json.load(f)
    logging.info(f"Loaded {len(ids)} ids from {ids_path}")

    for file_id in ids:
        url = URL_TEMPLATE.format(file_id)

        archive_file = TMP_DIR / f"{file_id}.7z"
        exported_file = TMP_DIR / f"{file_id}.npz"
        output_file = OUT_FOLDER / f"{file_id}.mp4"

        if output_file.exists():
            logging.info(f"Skipping {file_id} - already exists")
            continue

        if not archive_file.exists():
            gdown.download(url, output=str(TMP_DIR / f"{file_id}.7z"), quiet=False)

        logging.info(f"Extracting {file_id} to {TMP_DIR}")
        patoolib.extract_archive(
            str(archive_file), outdir=str(TMP_DIR), verbosity=1, interactive=False
        )
        archive_file.unlink()
        extracted_file = list(TMP_DIR.glob("*.hdf5"))[0]
        name_mapping[file_id] = extracted_file.stem
        with open(name_mapping_file, "w") as f:
            json.dump(name_mapping, f)

        logging.info("Starting export process".center(80, "="))
        subprocess.run(
            [
                "python",
                "-u",
                "export_h5.py",
                "--input",
                str(extracted_file),
                "--output",
                str(exported_file),
            ],
            check=True,
            text=True,
        )

        logging.info(f"Exported {file_id} to {exported_file}")
        extracted_file.unlink()
        logging.info("Starting conversion process".center(80, "="))
        subprocess.run(
            [
                "python",
                "-u",
                "convert_to_video.py",
                "--input",
                str(exported_file),
                "--output",
                str(output_file),
            ],
            check=True,
            text=True,
        )
        logging.info(f"Converted {file_id} to videos/{file_id}.mp4")
        exported_file.unlink()

    logging.info("Done".center(80, "="))
