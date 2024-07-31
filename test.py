from joblib import load

model_path = "models/model.joblib"
# C:\workFiles\DS\end_to_end\Project_Nature\project_nature\models\model.joblib
# model = tf.keras.models.load_model(model_path)
model = load(model_path)

print('hello')