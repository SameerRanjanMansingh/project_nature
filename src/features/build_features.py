import pathlib
import pandas as pd
import numpy as np
# from keras.preprocessing.image import ImageDataGenerator, img_to_array
from glob import glob
# from feature_defination import feature_build
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil


def data_generator(data_path):
    data =  ImageDataGenerator().flow_from_directory(data_path, target_size = (224,224))

    return data

from PIL import Image

def save_data(train_generator, test_generator, output_path):
    # Create output directories if they don't exist
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)
    
    # Save train images
    for i, (images, _) in enumerate(train_generator):
        for j, image_array in enumerate(images):
            image = Image.fromarray(image_array.astype('uint8'))
            filename = f"train_image_{i * train_generator.batch_size + j}.jpg"
            image.save(os.path.join(output_path, 'train', filename))
        if i * train_generator.batch_size >= len(train_generator.filenames):
            break
    
    # Save test images
    for i, (images, _) in enumerate(test_generator):
        for j, image_array in enumerate(images):
            image = Image.fromarray(image_array.astype('uint8'))
            filename = f"test_image_{i * test_generator.batch_size + j}.jpg"
            image.save(os.path.join(output_path, 'test', filename))
        if i * test_generator.batch_size >= len(test_generator.filenames):
            break
# def save_data(train, test, output_path):
#     # Save the split datasets to the specified output path
#     pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
#     train.to_csv(output_path + '/train.csv', index=False)
#     test.to_csv(output_path + '/test.csv', index=False)

if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    
    train_path = home_dir.as_posix() + '/data/raw/seg_train'
    test_path = home_dir.as_posix() + '/data/raw/seg_test'
    
    numberOfClass = len(glob(train_path + "/*"))

    train_data = data_generator(train_path)
    test_data = data_generator(test_path)

    output_path = home_dir.as_posix() + '/data/processed'

    # train_data = feature_build(train_data, 'train-data')
    # test_data = feature_build(test_data, 'test-data')

    save_data(train_data, test_data, output_path)


    