from template_model import ModelTemplate

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# https://keras.io/examples/cifar10_cnn/
class CNNModel(ModelTemplate):
    def __init__(self, input_shape):
        # fixed due to dataset
        num_classes = 10
        
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same',
                       input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

    
    def get_epochs(self):
        return 100

    def get_batch_size(self):
        return 8
    def get_optimizer(self):
        return keras.optimizers.SGD(lr=0.001, momentum=0.9)

    def get_loss_function(self):
        return 'categorical_crossentropy'

    def get_model(self):
        return self.model
    