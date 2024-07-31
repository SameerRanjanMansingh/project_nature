<<<<<<< HEAD
import sys
sys.path.append(r'c:\workFiles\DS\end_to_end\Project_Nature\project_nature')

=======
# train_model.py
>>>>>>> e480e4c03f94339652a018d65aa9a1048c69edd1
import pathlib
import sys
import joblib
import mlflow
from glob import glob
from box import ConfigBox
import os
import pandas as pd

from src.features.feature_defination import vgg16
<<<<<<< HEAD
from src.features.build_features import data_generator
=======
>>>>>>> e480e4c03f94339652a018d65aa9a1048c69edd1
# from src.common import read_yaml


def evaluate_model(train_data, test_data):
<<<<<<< HEAD
    try:
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
    except Exception as e:
        print("An error occurred during model evaluation:", str(e))

def save_model(model, output_path):
    try:
        # Save the trained model to the specified output path
        # joblib.dump(model, output_path + "/model.joblib")
        model.save(output_path + "/model.h5", save_format="h5")
    except Exception as e:
        print("An error occurred while saving the model:", str(e))


def main():
    try:
        curr_dir = pathlib.Path(__file__)
        home_dir = curr_dir.parent.parent.parent

        # input_file = sys.argv[1]
        # # data_path = home_dir.as_posix() + input_file

        output_path = home_dir.as_posix() + "/models"
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        train_path = home_dir.as_posix() + '/data/raw/seg_train'
        test_path = home_dir.as_posix() + '/data/raw/seg_test'
        

        # os.listdir(train_path)

        train_data = data_generator(train_path)
        test_data = data_generator(test_path)


        trained_model = evaluate_model(train_data, test_data)
        save_model(trained_model, output_path)
        # trained_model.save("model.h5")
        # We will push this model to S3 and also copy in the root folder for Dockerfile to pick
    except Exception as e:
        print("An error occurred in the main function:", str(e))
=======
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
>>>>>>> e480e4c03f94339652a018d65aa9a1048c69edd1


if __name__ == "__main__":
    main()
