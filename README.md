# DDD Scripts

A collection of utility scripts for processing [Davis Driving Dataset](https://sites.google.com/view/davis-driving-dataset-2020/home) data.

## `export_h5.py`

Exports the camera and polarity events from an `hdf5` recording. Output is in format of compressed numpy arrays:
- `frame_data` - _NxWxH_ numpy array with consecutive grayscale frames of the camera recording
- `frame_timestamps` - _N_ numpy array of timestamps of the corresponding frames
- `polarity_data` - _MxWxH_ numpy array with consecutive event camera activations (255 - positive, 0 - negative, 127 - neutral)
- `polarity_timestamps` - _M_ numpy array of timestamps of the corresponding 

## `voxelize_and_batch.py`

Processes the exported data from `export_*.py` output into a format viable for training.

## `convert_to_video.py`

Converts the exported data from `export_*.py` output into an `mp4` video with combined RGB (or GS) and event channels