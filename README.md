# 3D-RCAN

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

3D-RCAN is the companion code to our paper:

[Three-dimensional residual channel attention networks denoise and sharpen fluorescence microscopy image volumes](https://www.biorxiv.org/content/10.1101/2020.08.27.270439v1).

![Example](figures/example.png)

3D-RCAN is a 3D version of deep residual channel attention network (RCAN) [[1]](#1). This model is useful for restoring and enhancing volumetric time-lapse (4D) fluorescence microscopy data.

## System Requirements

---

- Windows 10. Linux and Mac OS should be able to run the code but the code has been only tested on Windows 10 so far.
- Python 3.6+
- NVIDIA GPU
- CUDA 10.0 and cuDNN 7.6.5

Tested Environment:

- Windows 10
- Python 3.7
- NVIDIA GTX 1060 6GB
- CUDA 10.0 and cuDNN 7.6.5

## Dependencies Installation

---

### (Option 1) Install dependencies in base environment

1. Download the `requirements.txt` from the repository
2. In your command prompt run:

    `pip install -r requirements.txt`

### (Option 2) Create a new virtual environment

1. Download the `requirements.txt` from the repository
2. Open command prompt and change directory to where you put the `requirements.txt`
3. Create a new virtual environment:

    `python -m venv RCAN3D`

4. Activate the virtual environment

    On Windows: `.\RCAN3D\Scripts\activate`
    On macOS and Linux: `source RCAN3D/bin/activate`

5. You should see (RCAN3D) in the command line.

6. In your command prompt run:

    `pip install -r requirements.txt`

## Training

---

We provide a sample dataset and a pretrained model for you to run the code, please download the files at
[here](https://www.dropbox.com/sh/hieldept1x476dw/AAC0pY3FrwdZBctvFF0Fx0L3a?dl=0).

To train the RCAN model, run:

`python train.py -c config.json -o /path/to/training/output/dir`

Training data is an array of raw and GT image pairs and it must be specified in the input config JSON file. Please check the example `config.json` in the repository. Following optional variables can be also set in the JSON file (if not set, default values will be used):

- validation_data (array of image pairs)
  - Validation data on which to evaluate the loss and metrics at the end of each epoch
  - Default: None

- epochs (integer)
  - Number of epochs to train the model
  - Default: 300

- steps_per_epoch (integer)
  - Number of steps to perform back-propagation on mini-batches in each epoch
  - Default: 256

- num_channels (integer)
  - Number of feature channels in RCAN
  - Default: 32

- num_residual_blocks (integer)
  - Number of residual channel attention blocks in each residual group in RCAN
  - Default: 3

- num_residual_groups (integer)
  - Number of residual groups in RCAN
  - Default: 5

- channel_reduction (integer)
  - Channel reduction ratio for channel attention
  - Default: 8

The default RCAN architecture is configured to be trained on a machine with 11GB GPU memory. If you encounter an OOM error during training, please try reducing model parameters such as `num_residual_blocks` and `num_residual_groups`. In the example `config.json`, we reduce `num_residual_groups` to 3 to run on a 6GB GTX 1060 GPU.

The loss values are saved in the training output folder. You can use TensorBoard to monitor the loss values. To use TensorBoard, run the following command and open [http://127.0.0.1:6006] in your browser.

`tensorboard --host=127.0.0.1 --logdir=/path/to/training/dir`

## Model Apply

---

We provide two ways to apply trained 3D-RCAN models

### (Option 1) Apply model to one image at a time

To apply the trained model to an image, run:

`python apply.py -m /path/to/training/output/dir -i input_raw_image.tif -o output.tif`

The best model (i.e. the one with the lowest loss) will be selected from the model directory and applied. The output TIFF file is a two-channel ImageJ Hyperstack containing raw and restored images.

### (Option 2) Apply model to a folder of images

You can turn on the “batch apply” mode by passing a directory path to the “-i” argument, e.g.:

`python apply.py -m /path/to/training/output/dir -i /path/to/input/image/dir -o /path/to/output/image/dir`

When the input (specified by “-i”) is a directory, the output (“-o”) must be a directory too. The output directory is created by the script if it doesn’t exist yet.

You can also specify a directory where ground truth images are located. The ground truth directory must contain the same number of images as the input directory.

`python apply.py -m model_dir -i input_dir -g ground_truth_dir -o output_dir`

Following two more arguments are available:

- `-g` or `--ground_truth`

    Reference ground truth image. If it is set, the output TIFF is a three-channel ImageJ Hyperstack with raw, restored, and GT
- `-b` or `--bpp`

    Bit depth of the output image (either 8, 16, or 32). If not specified, it will be set to 32

## References

---

<a id="1">[1]</a>
Yulun Zhang *et al.* (2018).
Image Super-Resolution Using Very Deep
Residual Channel Attention Networks.
ECCV 2018

## License

---
- Copyright 2020 DRVision Technologies LLC.
- [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) International Public License](https://creativecommons.org/licenses/by-nc/4.0/) 