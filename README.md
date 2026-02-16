# Kadaster Code

Code for processing Kadaster/BRT data and downloading images.

## Requirements

- Python 3.10+
- pip

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

1. Install dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

## Usage

### Download images

```bash
python download_images.py
```

This downloads 625 images from the JSON file `zutphen met labels` to the output folder.

### Process images

```bash
python process_images.py
```

This processes downloaded images into numpy arrays for training.

### Train model

```bash
python train_model.py
```

This trains an autoencoder model on the processed images.
