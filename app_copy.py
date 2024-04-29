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


# os.putenv('LANG', 'en_US.UTF-8')
# os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)



def predict_image(filename):
    model_path = "models/model.h5"
    model = tf.keras.models.load_model(model_path)

    img_ = load_img(filename, target_size=(228, 228))
    img_array = img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)
    
    predicted_class = np.argmax(prediction)
    
    return predicted_class

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename) #predict_image(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) #local host
    # app.run(host='0.0.0.0', port=8080) #for AWS
    # app.run(host='0.0.0.0', port=80) #for AZURE

