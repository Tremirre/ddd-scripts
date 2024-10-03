import argparse
import logging
import pathlib

import tqdm
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        help="Path to the input file",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Path to the output file",
        default="output.mp4",
    )

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    arrs = np.load(input_file)

    frames_data = arrs["frame_data"]
    polarities_data = arrs["polarity_data"]
    polarity_groups = arrs["polarity_groups"]

    logging.info(f"Loaded {input_file} successfully.")
    logging.info(f"Frames: {len(frames_data):>10}")
    logging.info(f"Polarities: {len(polarities_data):>10}")

    logging.info("Averaging polarities...")
    averaged_polarities = []
    for i in tqdm.tqdm(range(len(frames_data))):
        mask = polarity_groups == i
        if np.any(mask):
            averaged_polarity = np.mean(polarities_data[mask], axis=0)
            averaged_polarities.append(averaged_polarity)
        else:
            averaged_polarities.append(np.ones_like(polarities_data[0]) * 128)

    del polarities_data, polarity_groups

    averaged_polarities = np.array(averaged_polarities, dtype=np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

    logging.info(f"Saving video as {output_file}...")
    out = cv2.VideoWriter(str(output_file), fourcc, 60.0, (346, 260 * 2), 0)

    for frame, polarity in tqdm.tqdm(
        zip(frames_data, averaged_polarities), total=len(frames_data)
    ):
        frame = frame.astype(np.uint8)
        frame = np.vstack([frame, polarity])
        out.write(frame)

    out.release()
    logging.info("Done!")
