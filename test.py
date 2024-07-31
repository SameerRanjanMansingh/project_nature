<<<<<<< HEAD
from joblib import load

model_path = "models/model.joblib"
# C:\workFiles\DS\end_to_end\Project_Nature\project_nature\models\model.joblib
# model = tf.keras.models.load_model(model_path)
model = load(model_path)

print('hello')
=======
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from common import decodeImage
# from src.cnnClassifier.pipeline.predict import PredictionPipeline
from joblib import load
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
from src.pipeline.predict import PredictionPipeline



filename = "inputImage.jpg"
model_path = "models/model.h5"
model = tf.keras.models.load_model(model_path)

img_ = load_img(filename, target_size=(228, 228))
img_array = img_to_array(img_)
img_processed = np.expand_dims(img_array, axis=0)
img_processed /= 255.

prediction = model.predict(img_processed)

result = np.argmax(prediction)

# print(predicted_class)

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
>>>>>>> e480e4c03f94339652a018d65aa9a1048c69edd1
