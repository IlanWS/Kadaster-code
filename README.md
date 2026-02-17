# Kadaster Code

Code for processing Kadaster/BRT data and downloading images.

## Requirements

- Python 3.12.x
- pip

## Setup

### Standard Setup (CPU only)

1. Create and activate a virtual environment with Python 3.12:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -e .
   ```

### AMD GPU Support (ROCm)

If you have an AMD GPU with ROCm 4+ installed, the setup will automatically detect it and install the ROCm-compatible TensorFlow packages from the [ROCm repository](https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/).

ROCm 4+ must be installed on your system for this to work.

## Usage

### Download images

```bash
python -m kadaster_code download
```

This downloads 625 images from the JSON file `zutphen met labels` to the output folder.

### Process images

```bash
python -m kadaster_code process
```

This processes downloaded images into numpy arrays for training.

### Train model

```bash
python -m kadaster_code train
```

This trains an autoencoder model on the processed images.
