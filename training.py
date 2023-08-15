import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import cv2 
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten
from keras.utils import to_categorical


directory = 'dataset/'
no_tumor_data = os.listdir(directory + 'no/')
yes_tumor_data = os.listdir(directory + 'yes/')
data = []
tag = []

input_size = 64 

# Load and resize images directly using PIL
for name in no_tumor_data:
    if name.split('.')[1] == 'jpg':
        picture = cv2.imread(directory + 'no/' + name)
        picture = Image.fromarray(picture, 'RGB')
        picture = picture.resize((input_size, input_size))
        data.append(np.array(picture))
        tag.append(0)

for name in yes_tumor_data:
    if name.split('.')[1] == 'jpg':
        picture = cv2.imread(directory + 'yes/' + name)
        picture = Image.fromarray(picture, 'RGB')
        picture = picture.resize((input_size, input_size))
        data.append(np.array(picture))
        tag.append(1)


data = np.array(data)
tag = np.array(tag)

x_train, x_test, y_train, y_test = train_test_split(data, tag, test_size=0.2, random_state=0)



# print(x_test.shape)
# print(y_test.shape)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train=to_categorical(y_train, num_classes=2)
y_test=to_categorical(y_test, num_classes=2)


s_model = Sequential()

s_model.add(Conv2D(32, (3,3),input_shape=(input_size, input_size, 3)))
s_model.add(Activation('relu'))
s_model.add(MaxPooling2D(pool_size=(2,2)))

s_model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
s_model.add(Activation('relu'))
s_model.add(MaxPooling2D(pool_size=(2,2)))

s_model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
s_model.add(Activation('relu'))
s_model.add(MaxPooling2D(pool_size=(2,2)))

s_model.add(Flatten())
s_model.add(Dense(64))
s_model.add(Activation('relu'))
s_model.add(Dropout(0.5))
s_model.add(Dense(2))
s_model.add(Activation('softmax'))


s_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

s_model.fit(x_train, y_train, batch_size=20, verbose=1, epochs=30, validation_data=(x_test, y_test), shuffle=False)

s_model.save('10epochTumordetectorCategorical.h5')