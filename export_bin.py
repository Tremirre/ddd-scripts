from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import cv2  # type: ignore
import numpy as np  # type: ignore
import tqdm  # type: ignore
import tqdm.contrib.logging  # type: ignore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclasses.dataclass
class Config:
    input_bin: pathlib.Path
    input_vid: pathlib.Path
    output: pathlib.Path

    @classmethod
    def from_args(cls):
        args = argparse.ArgumentParser()
        args.add_argument(
            "--input-vid",
            type=pathlib.Path,
            help="Path to the input video file",
        )
        args.add_argument(
            "--input-bin",
            type=pathlib.Path,
            help="Path to the input binary file",
        )
        args.add_argument(
            "--output",
            type=pathlib.Path,
            help="Path to the output file",
            default="output.npz",
        )
        return cls(**vars(args.parse_args()))


if __name__ == "__main__":
    config = Config.from_args()

    if not config.input_bin.exists():
        raise FileNotFoundError(f"File {config.input_bin} not found - exiting.")

    if not config.input_vid.exists():
        raise FileNotFoundError(f"File {config.input_vid} not found - exiting")

    with open(config.input_bin, "rb") as f:
        height, width, count = np.fromfile(f, dtype=np.uint32, count=3)
        polarity_data: np.ndarray = np.fromfile(
            f, dtype=np.uint8, count=height * width * count
        ).reshape((count, height, width))
        polarity_timestmaps: np.ndarray = np.fromfile(f, dtype=np.float32, count=count)

    polarity_timestmaps -= polarity_timestmaps[0]
    cap = cv2.VideoCapture(str(config.input_vid))
    timestamps = []
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamps.append(ts)
        frames.append(frame)

    frames = np.array(frames)
    timestamps = np.array(timestamps)

    assert timestamps.shape[0] == polarity_timestmaps.shape[0]  # type: ignore

    polarity_groups = np.searchsorted(timestamps, polarity_timestmaps)  # type: ignore
    max_groups = polarity_groups.max()
    assert max_groups < 2**16, "Too many groups for uint16"

    out_data = {
        "polarity_groups": polarity_groups.astype(np.uint16),
        "polarity_data": polarity_data,
        "frame_data": frames,
    }
    logging.info(f"Saving data to {config.output}")
    np.savez_compressed(config.output, **out_data)
    logging.info("DONE!")
