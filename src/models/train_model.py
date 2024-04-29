# train_model.py
import pathlib
import sys
import joblib
import mlflow
from glob import glob
from box import ConfigBox
import os
import pandas as pd

from src.features.feature_defination import vgg16
# from src.common import read_yaml


def evaluate_model(train_data, test_data):
    # param = ConfigBox('params.yaml')

    # epochs = param.EPOCHS
    # numberOfClass = param.CLASSES
    # vgg16Model = vgg16(numberOfClass)

    # # Traning with model
    # batch_size = param.BATCH_SIZE

    epochs = 2
    numberOfClass = 6
    vgg16Model = vgg16(numberOfClass)

    # Traning with model
    batch_size = 32

    model = vgg16Model.fit(train_data, 
                           steps_per_epoch = 1600 // batch_size, 
                           epochs = epochs, 
                           validation_data = test_data, 
                           validation_steps = 800 // batch_size)

    return model

def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + "/model.joblib")


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    # input_file = sys.argv[1]
    # # data_path = home_dir.as_posix() + input_file

    output_path = home_dir.as_posix() + "/models"
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    
    train_path = home_dir.as_posix() + '/data/processed/train'
    test_path = home_dir.as_posix() + '/data/processed/test'

    # os.listdir(train_path)

    train_data = os.listdir(train_path)
    test_data = os.listdir(test_path)


    trained_model = evaluate_model(train_data, test_data)
    # save_model(trained_model, output_path)
    trained_model.save("model.h5")
    # We will push this model to S3 and also copy in the root folder for Dockerfile to pick


if __name__ == "__main__":
    main()
