import os
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_FILE'] = 'OCR_Resnet.h5'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

model = load_model(app.config['MODEL_FILE'], compile=False)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  
    img = img.resize((32, 32))
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)  
    img_array = img_array / 255.0 
    return img_array

def predict_letter(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    predicted_letter = chr(ord('A') + predicted_class)
    return predicted_letter

@app.route("/")
def index():
    return "Hello World!"

@app.route("/predict", methods=["POST"])
def predict_letter_route():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            predicted_letter = predict_letter(image_path)

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting"
                },
                "data": {
                    "predicted_letter": predicted_letter,
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Invalid file format. Please upload a JPG, JPEG, or PNG image."
                },
                "data": None,
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405

if __name__ == "_main_":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))