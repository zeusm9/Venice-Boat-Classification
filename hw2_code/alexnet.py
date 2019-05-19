
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras import backend as K

np.random.seed(78)

class Alexnet:
    @staticmethod
    def build(width,height,depth,classes,finalAct="softmax"):
        model = Sequential()
        inputShape = (height,width,depth)
        chanDim = -1
        K.tensorflow_backend._get_available_gpus()
        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1

        model = Sequential()
        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=inputShape, kernel_size=(11, 11),
                         strides=(4, 4), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Passing it to a dense layer
        model.add(Flatten())
        # 1st Dense Layer
        model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        #model.add(Dropout(0.25))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 2nd Dense Layer
        model.add(Dense(1024))
        model.add(Activation('relu'))
        # Add Dropout
        #model.add(Dropout(0.25))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Dense Layer
        model.add(Dense(1024))
        model.add(Activation('relu'))
        # Add Dropout
        #model.add(Dropout(0.25))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Output Layer
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return  model

