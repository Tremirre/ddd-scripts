import argparse
import logging
import pathlib

import tqdm
import numpy as np
import scipy.interpolate  # type: ignore

import reader


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

S_H, S_W = (260, 346)
T_H, T_W = (256, 336)

OFFSET_H = (S_H - T_H) // 2
OFFSET_W = (S_W - T_W) // 2

CHUNK_SIZE = 8192
BATCH_SIZE = 128


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        help="Path to the input file",
        required=True,
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=6,
        help="Number of bins to use for the voxelization",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
        default=False,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    args = parser.parse_args()
    input_file = args.input
    output_dir = args.output
    n_bins = args.num_bins
    b_size = args.batch_size

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")

    logging.debug(f"Input file: {input_file}")
    logging.debug(f"Output dir: {output_dir}")
    logging.debug(f"Number of bins: {n_bins}")
    logging.debug(f"Chunk size: {CHUNK_SIZE}")

    target_times = np.linspace(0, 1, n_bins)
    frames = None
    polarities = None
    with reader.ZipNumpyReader(input_file) as npz_file:
        frames_iter = npz_file.get_iterator("frame_data", CHUNK_SIZE)
        polarities_iter = npz_file.get_iterator("polarity_data", CHUNK_SIZE)
        polarity_groups = npz_file.get_array("polarity_groups")
        num_frames = polarity_groups.max() + 1
        logging.info(f"Found {num_frames} frames in the input file.")
        num_polarities = len(polarity_groups)
        logging.info(f"Found {num_polarities} polarities in the input file.")

        voxelled_polarities = np.zeros((num_frames, n_bins, T_H, T_W), dtype=np.float16)
        trimmed_frames = np.zeros((num_frames, T_H, T_W), dtype=np.uint8)
        logging.info(f"Loaded {input_file} successfully.")
        logging.info("Processing data in chunks...")
        prev_group = -1
        arr_offset = 0
        for i, polarities in enumerate(polarities_iter):
            polarities = polarities[
                :, OFFSET_H : S_H - OFFSET_H, OFFSET_W : S_W - OFFSET_W
            ].astype(np.float16)
            polarities = (polarities - 127.5) / 127.5
            groups = polarity_groups[arr_offset : arr_offset + len(polarities)]
            logging.debug(
                f"Groups: {groups[:10]}...{groups[-10:]}, Offset:{arr_offset} - {arr_offset + len(polarities)}"
            )
            arr_offset += len(polarities)
            min_group = np.min(groups)
            prev_group = np.max(groups)
            group_range = range(min_group, prev_group + 1)
            logging.info(
                f"Processing polarities - chunk {i+1:>3}. - polarities: {len(polarities):>10} (frames {min_group} - {prev_group})"
            )

            for i in tqdm.tqdm(
                group_range, desc="Voxelizing polarities", total=len(group_range)
            ):
                if i == prev_group:
                    continue

                mask = groups == i
                if not np.any(mask):
                    continue

                polarities_chunk = polarities[mask]
                if len(polarities_chunk) == 1:
                    voxel = np.repeat(polarities_chunk, n_bins, axis=0)
                else:
                    original_times = np.linspace(0, 1, np.sum(mask))
                    interpolator = scipy.interpolate.interp1d(
                        original_times, polarities_chunk, kind="linear", axis=0
                    )
                    voxel = interpolator(target_times)
                voxelled_polarities[i] = voxel

        for i, frames in enumerate(frames_iter):
            logging.info(
                f"Processing frames- chunk {i+1:>3}. - frames: {len(frames):>10}"
            )
            left = i * CHUNK_SIZE
            right = left + len(frames)
            logging.debug(f"Frames: {left} - {right}")
            trimmed_frames[left:right] = frames[
                :, OFFSET_H : S_H - OFFSET_H, OFFSET_W : S_W - OFFSET_W
            ]
        logging.debug("Cleaning up memmapped arrays")
        del polarities
        del frames

    n_batches, rem_batches = divmod(num_frames, b_size)
    n_batches += bool(rem_batches)

    logging.info(f"Saving data to {output_dir} in {n_batches} batches...")
    for i in tqdm.tqdm(range(0, len(voxelled_polarities), b_size)):
        np.savez(
            output_dir / f"{input_file.stem}_{i // b_size:>04}.npz",
            frame_data=trimmed_frames[i : i + b_size],
            polarity_data=voxelled_polarities[i : i + b_size],
        )
    logging.info("Data saved successfully.")
