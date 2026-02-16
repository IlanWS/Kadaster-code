import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from tensorflow.keras.optimizers import Adam

input_image_array = np.empty([625,512,512,3], dtype=np.int16)
target_image_array = np.empty([625,512,512,3], dtype=np.int16)

for i in range(625):
    image_path1 = os.path.join("zutphen zonder labels map", f"image_{i}.jpg")
    image1 = Image.open(image_path1)
    image_array1 = np.array(image1)
    input_image_array[i] = np.delete(np.delete(np.delete(image_array1, np.s_[512::], 0),np.s_[512::], 1),np.s_[3::], 2)

    image_path2 = os.path.join("zutphen met alleen labels map", f"image_{i}.jpg")
    image2 = Image.open(image_path2)
    image_array2 = np.array(image2)
    target_image_array[i] = np.delete(np.delete(np.delete(image_array2, np.s_[512::], 0),np.s_[512::], 1),np.s_[3::], 2)

x_train, x_test, y_train, y_test = train_test_split(input_image_array, target_image_array, test_size=0.2, random_state=42)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

def create_model(img_width, img_height):
    x = Input(shape=(img_width, img_height, 3))
    e_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    pool1 = MaxPooling2D((2, 2), padding='same')(e_conv1)
    batchnorm_1 = BatchNormalization()(pool1)
    e_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(batchnorm_1)
    pool2 = MaxPooling2D((2, 2), padding='same')(e_conv2)
    batchnorm_2 = BatchNormalization()(pool2)
    e_conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(batchnorm_2)
    h = MaxPooling2D((2, 2), padding='same')(e_conv3)
    d_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
    up1 = UpSampling2D((2, 2))(d_conv1)
    d_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2, 2))(d_conv2)
    d_conv3 = Conv2D(16, (3, 3), activation='relu', padding="same")(up2)
    up3 = UpSampling2D((2, 2))(d_conv3)
    r = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3)
    model = Model(x, r)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model

model = create_model(512, 512)
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=4, validation_data=(x_test, y_test))
