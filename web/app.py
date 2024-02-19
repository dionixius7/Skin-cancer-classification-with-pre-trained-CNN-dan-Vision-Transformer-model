from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import urllib.parse

app = Flask(__name__)

model_path = "E:/Skripsi Dion/FIX/web/skin_cancer_HAM10000_VGG16.hdf5"
model = load_model(model_path)

classname = [
    "akiec: Actinic keratoses",
    "bcc: Basal cell carcinoma",
    "bkl: Benign keratosis",
    "df: Dermatofibroma",
    "mel: Melanoma",
    "nv: Melanocytic nevi",
    "vasc: Vascular lesions",
]

app.config["UPLOAD_FOLDER"] = "E:/Skripsi Dion/FIX/image"


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        image_file = request.files["imagefile"]
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
        image_file.save(image_path)

        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = classname[predicted_class_index]
        accuracy = predictions[0][predicted_class_index]

        result = {
            "Jenis": predicted_class,
            "Akurasi": accuracy.item(),
        }

        img_base_name = os.path.basename(image_path)

        return render_template("output.html", img=img_base_name, result=result)


@app.route("/<fileimg>")
def send_uploaded_image(fileimg=""):
    decoded_path = urllib.parse.unquote(fileimg)
    return send_from_directory(app.config["UPLOAD_FOLDER"], decoded_path)


if __name__ == "__main__":
    app.run(port=8000, debug=True)
