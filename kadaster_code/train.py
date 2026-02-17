import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if os.environ.get("ROCM_PATH"):
    os.environ["TF_ROCM_AMD_GPU_BUILD"] = "1"
    os.environ["TF_ROCM_ENABLE_XLA"] = "1"

import numpy as np
from sklearn.model_selection import train_test_split

from kadaster_code.data.preprocessor import process_images
from kadaster_code.models.autoencoder import create_model


def train(
    input_folder: str = "zutphen-zonder-labels-map",
    target_folder: str = "zutphen-met-alleen-labels-map",
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 10,
    batch_size: int = 4,
    img_width: int = 512,
    img_height: int = 512,
) -> None:
    """Train autoencoder model on processed images."""
    input_array, target_array = process_images(input_folder, target_folder)

    x_train, x_test, y_train, y_test = train_test_split(
        input_array, target_array, test_size=test_size, random_state=random_state
    )

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    model = create_model(img_width=img_width, img_height=img_height)
    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
    )
