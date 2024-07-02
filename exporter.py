from __future__ import annotations

import argparse
import dataclasses
import enum
import logging
import pathlib
import struct

import tqdm
import tqdm.contrib.logging
import numpy as np
import h5py

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

IMAGE_SHAPE = (260, 346)


class EventType(enum.IntEnum):
    SPECIAL = 0
    POLARITY = 1
    FRAME = 2
    IMU6 = 3
    IMU9 = 4


@dataclasses.dataclass
class EventHeader:
    type: EventType
    source: int
    size: int
    offset: int
    overflow: int
    capacity: int
    number: int
    valid: int

    @classmethod
    def from_buffer(cls, buffer: bytes) -> EventHeader:
        return cls(*struct.unpack("hhiiiiii", buffer))


@dataclasses.dataclass
class ParsedEvent:
    timestamp: float
    data: np.ndarray


def parse_frame_body(body: np.ndarray, header: EventHeader) -> ParsedEvent:
    sub_header = body[:36].view(np.uint32)
    timestamp = sub_header[2] * 1e-6
    sub_body = (body[36:].view(np.uint16) / 256).astype(np.uint8)
    img = sub_body.reshape(*IMAGE_SHAPE)[::-1, ::-1]
    return ParsedEvent(timestamp, img)


def parse_polarity_body(body: np.ndarray, header: EventHeader) -> ParsedEvent:
    p_arr = body.view(np.uint32).reshape((header.capacity, header.size // 4))
    data, timestamp = p_arr[:, 0], p_arr[:, 1]
    pol = data >> 1 & 0b1
    y = data >> 2 & 0b111111111111111
    x = data >> 17
    polarity_frame = np.ones(IMAGE_SHAPE, dtype=np.uint8) * 127
    polarity_frame[y, x] = pol * 255
    return ParsedEvent(timestamp[0] * 1e-6, polarity_frame)


PARSERS = {
    EventType.FRAME: parse_frame_body,
    EventType.POLARITY: parse_polarity_body,
}


@dataclasses.dataclass
class Config:
    input: pathlib.Path
    output: pathlib.Path

    @classmethod
    def from_args(cls):
        args = argparse.ArgumentParser()
        args.add_argument(
            "--input",
            type=pathlib.Path,
            help="Path to the input file",
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

    try:
        source = h5py.File(config.input, "r")
    except OSError:
        logging.error(f"Failed to open {config.input} - exiting.")
        exit(1)
    logging.info(f"Opened {config.input} successfully.")
    dvs_data = source["dvs"]["data"]

    event_aggregates = {
        EventType.FRAME: {"timestamps": [], "data": []},
        EventType.POLARITY: {"timestamps": [], "data": []},
    }
    total_events = len(dvs_data)
    for i, event in tqdm.tqdm(enumerate(dvs_data), total=total_events):
        with tqdm.contrib.logging.logging_redirect_tqdm():
            sys_ts, header, body = event
            try:
                parsed_header = EventHeader.from_buffer(header)
            except struct.error:
                logging.error(f"Failed to parse event {i+1} header - skipping.")
                continue
            parser = PARSERS.get(parsed_header.type)
            if parser is None:
                logging.debug(
                    f"Unknown event type: {parsed_header.type} - skipping event {i+1}."
                )
                continue
            event = parser(body, parsed_header)
            event_aggregates[parsed_header.type]["timestamps"].append(event.timestamp)
            event_aggregates[parsed_header.type]["data"].append(event.data)

    out_data = {}
    for etype, data in event_aggregates.items():
        out_data[f"{etype.name.lower()}_timestamps"] = np.array(
            data["timestamps"], dtype=np.float32
        )
        del data["timestamps"]
        out_data[f"{etype.name.lower()}_data"] = np.array(data["data"], dtype=np.uint8)
        del data["data"]

    logging.info(f"Saving data to {config.output}")
    np.savez_compressed(config.output, **out_data)
    logging.info("DONE!")
