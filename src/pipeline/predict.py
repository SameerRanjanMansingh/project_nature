import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf
from joblib import load
from keras.preprocessing.image import load_img, img_to_array



class PredictionPipeline:
    def __init__(self, filename):
        self.filename=filename

    def predict(self):
        # model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # imagename = self.filename
        # test_image = image.load_img(imagename, target_size=(224,224))
        # test_image = image.img_to_array(test_image)
        # test_image = np.expand_dims(test_image, axis=0)
        # result = np.argmax(model.predict(test_image), axis=1)
        # print(result)

        # if result[0] == 1:
        #     prediction = 'Healthy'
        #     return [{"image": prediction}]
        # else:
        #     prediction = 'Coccidiosis'
        #     return [{"image": prediction}]
        model_path = "models/model.h5"
        model = tf.keras.models.load_model(model_path)

        img_ = load_img(self.filename, target_size=(228, 228))
        img_array = img_to_array(img_)
        img_processed = np.expand_dims(img_array, axis=0)
        img_processed /= 255.

        prediction = model.predict(img_processed)
        
        result = np.argmax(prediction)
        
        if result == 0:
            prediction = 'Buildings'
            return [{"image": prediction}]
        elif result == 1:
            prediction = 'Forest'
            return [{"image": prediction}]
        elif result == 2:
            prediction = 'Glacier'
            return [{"image": prediction}]
        elif result == 3:
            prediction = 'Mountain'
            return [{"image": prediction}]
        elif result == 4:
            prediction = 'Sea'
            return [{"image": prediction}]
        else:
            prediction = 'Street'
            return [{"image": prediction}]
        # return predicted_class