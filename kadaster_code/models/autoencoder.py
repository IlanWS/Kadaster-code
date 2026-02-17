from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
)
from tensorflow.keras.optimizers import Adam


def create_model(
    img_width: int = 512, img_height: int = 512, learning_rate: float = 0.0005
) -> Model:
    """Create autoencoder model for image processing."""
    x = Input(shape=(img_width, img_height, 3))
    
    e_conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    pool1 = MaxPooling2D((2, 2), padding="same")(e_conv1)
    batchnorm_1 = BatchNormalization()(pool1)
    
    e_conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(batchnorm_1)
    pool2 = MaxPooling2D((2, 2), padding="same")(e_conv2)
    batchnorm_2 = BatchNormalization()(pool2)
    
    e_conv3 = Conv2D(16, (3, 3), activation="relu", padding="same")(batchnorm_2)
    h = MaxPooling2D((2, 2), padding="same")(e_conv3)
    
    d_conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(h)
    up1 = UpSampling2D((2, 2))(d_conv1)
    
    d_conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(up1)
    up2 = UpSampling2D((2, 2))(d_conv2)
    
    d_conv3 = Conv2D(16, (3, 3), activation="relu", padding="same")(up2)
    up3 = UpSampling2D((2, 2))(d_conv3)
    
    r = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(up3)
    
    model = Model(x, r)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    
    return model
