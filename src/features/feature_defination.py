import pathlib
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19




def vgg16(numberOfClass):
    # Import model
    vgg16 = VGG16()
    vgg16_layer_list = vgg16.layers

    # add the layers of vgg16 in my created model.
    vgg16Model = Sequential()
    for i in range(len(vgg16_layer_list)-1):
        vgg16Model.add(vgg16_layer_list[i])


    # Close the layers of vgg16
    for layers in vgg16Model.layers:
        layers.trainable = False

    # Last layer
    vgg16Model.add(Dense(numberOfClass, activation = "softmax"))

    vgg16Model.compile(loss = "categorical_crossentropy", optimizer = "rmsprop",metrics = ["accuracy"])
    
    return vgg16Model


if __name__ == '__main__':
    # curr_dir = pathlib.Path(__file__)
    # home_dir = curr_dir.parent.parent.parent
    # data_path = home_dir.as_posix() + '/data/raw/test.csv'

    # curr_dir = pathlib.Path(__file__)
    # home_dir = curr_dir.parent.parent.parent
    
    # train_path = home_dir.as_posix() + '/data/raw/seg_train'
    # test_path = home_dir.as_posix() + '/data/raw/seg_test'
    

    # train_data = data_generator(train_path)
    # test_data = data_generator(test_path) 
    pass