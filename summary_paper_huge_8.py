from template_model import ModelTemplate

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

#https://arxiv.org/pdf/1611.04905.pdf
class CNNModel(ModelTemplate):
    def __init__(self, input_shape):
        # fixed due to dataset
        num_classes = 10
        
        self.model = Sequential()
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2), strides=2))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2), strides=2))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2), strides=2))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2), strides=2))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))
    
    def get_epochs(self):
        return 10

    def get_batch_size(self):
        return 8
    def get_optimizer(self):
        return keras.optimizers.SGD(lr=0.001, momentum=0.9)

    def get_loss_function(self):
        return 'categorical_crossentropy'

    def get_model(self):
        return self.model
    