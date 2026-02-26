import tensorflow as tf
from tensorflow.python.keras import layers, models, optimizers
import numpy as np
from Data_preprocessing import *
from Hyperparameters import *

#import keras
#from keras import layers, models, optimizers
#from Data_preprocessing import *

def build_model(heigth, width, channels):
    inputs = layers.Input((heigth, width, channels))

    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    # encoder
    f1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(f1)

    f2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(f2)

    f3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(f3)

    # bottleneck
    bottleneck = conv_block(p3, 512)

    # decoder ("concatenate" represents the skip connections with the encoder blocks, see architecture)
    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    u3 = layers.concatenate([u3, f3])
    f4 = conv_block(u3, 256)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(f4)
    u2 = layers.concatenate([u2, f2])
    f5 = conv_block(u2, 128)

    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(f5)
    u1 = layers.concatenate([u1, f1])
    f6 = conv_block(u1, 64)

    # sigmoid to get float values between 0 and 1
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(f6)

    model = models.Model(inputs, outputs, name="U-Net_RoadLabeler")
    return model

def compile_model():
    #number of channels of the input is 1 (binary image), parameter should be changed when working with colour images (channels = 3 for RGB image)
    model = build_model(512,512,1)

    #mse might not be the most relevant metric, but we cannot use metrics such as accuracy or recall, as the output is not in binary,like the input is
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.BinaryFocalCrossentropy(), metrics=['mae'])
    print(model.summary())
    return model

def train_model():
    x_train, y_train, x_test, y_test = data_split()
    model = compile_model()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    prediction = np.squeeze(model.predict(x_test))
    return prediction

