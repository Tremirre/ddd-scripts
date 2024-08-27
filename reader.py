import logging
import typing
import zipfile
import tempfile

import numpy as np


class NumpyMemmapIterator:
    def __init__(
        self,
        array_file: str,
        chunk_size: int,
        dtype: type,
        full_shape: tuple[int],
    ) -> None:
        self.array_file = array_file
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.offset = 0
        self.full_shape = full_shape

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        chunk_size = min(self.chunk_size, self.full_shape[0] - self.offset)
        if chunk_size <= 0:
            raise StopIteration

        chunk_shape = (chunk_size,)
        if len(self.full_shape) > 1:
            chunk_shape = (chunk_size, *self.full_shape[1:])
        logging.debug(f"Reading chunk with shape {chunk_shape}")
        chunk_array = np.lib.format.open_memmap(
            self.array_file,
            dtype=self.dtype,
            mode="c",
            shape=self.full_shape,
        )
        chunk_array = chunk_array[self.offset : self.offset + chunk_size]
        self.offset += chunk_size
        return chunk_array


class ZipNumpyReader:
    def __init__(self, input_file: str) -> None:
        self.input_file = input_file
        self.array_metadata = {}
        with zipfile.ZipFile(self.input_file, "r") as archive:
            for file in archive.namelist():
                if not file.endswith(".npy"):
                    continue
                npy = archive.open(file)
                version = np.lib.format.read_magic(npy)
                shape, _, dtype = np.lib.format._read_array_header(npy, version)
                logging.debug(
                    f"Found array {file} with shape {shape} and dtype {dtype}"
                )
                self.array_metadata[file] = {"shape": shape, "dtype": dtype}

            self.tmp_dir = tempfile.TemporaryDirectory(prefix="NPZR_")
            logging.info(f"Extracting {self.input_file} to {self.tmp_dir.name}...")
            archive.extractall(self.tmp_dir.name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.info(f"Cleaning up {self.tmp_dir.name}")
        self.tmp_dir.cleanup()

    def get_iterator(
        self, arr_name: str, chunk_size: int, dtype: type | None = None
    ) -> NumpyMemmapIterator:

        return NumpyMemmapIterator(
            f"{self.tmp_dir.name}\\{arr_name}.npy",
            chunk_size,
            dtype or self.array_metadata[f"{arr_name}.npy"]["dtype"],
            self.array_metadata[f"{arr_name}.npy"]["shape"],
        )

    def get_array(self, arr_name: str) -> np.ndarray:
        return np.load(f"{self.tmp_dir.name}\\{arr_name}.npy")

    def get_metadata(self, arr_name: str) -> dict:
        return self.array_metadata[f"{arr_name}.npy"]
