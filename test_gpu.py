#!/usr/bin/env python3
"""Test script to verify TensorFlow ROCm GPU detection."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import tensorflow as tf

    print(f"TensorFlow version: {tf.__version__}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"✓ GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")

        tf.config.set_visible_devices(gpus[0], "GPU")
        print("✓ TensorFlow can access GPU")
    else:
        print("✗ No GPUs detected by TensorFlow")
        print("  This may indicate ROCm is not properly installed or configured")

    logical_gpus = tf.config.list_logical_devices("GPU")
    if logical_gpus:
        print(f"Logical GPUs: {len(logical_gpus)}")

except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Please install tensorflow-rocm first")
