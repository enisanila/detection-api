import os
import numpy as np
from flask import Flask, jsonify, request
# from PIL import Image
from keras.models import load_model
from werkzeug.utils import secure_filename
import cv2
import imutils
from imutils.contours import sort_contours

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_FILE'] = 'OCR_Resnet.h5'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

model = load_model(app.config['MODEL_FILE'], compile=False)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_edged = cv2.Canny(img_blurred, 30, 150)
    img_contours = cv2.findContours(img_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = imutils.grab_contours(img_contours)
    img_contours = sort_contours(img_contours, method="left-to-right")[0]
    chars = []

    for c in img_contours:
        (x, y, w, h) = cv2.boundingRect(c)

        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            img_roi = img_gray[y:y+h, x:x+w]
            thresh = cv2.threshold(
                img_roi, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            (tH, tW) = thresh.shape

            if tW > tH:
                thresh = imutils.resize(thresh, width=32)
            else:
                thresh = imutils.resize(thresh, height=32)

            (tH, tW) = thresh.shape
            dX = int(max(0, 32-tW) / 2.0)
            dY = int(max(0, 32-tH) / 2.0)

            img_padded = cv2.copyMakeBorder(thresh,
                                            top=dY,
                                            bottom=dY,
                                            left=dX,
                                            right=dX,
                                            borderType=cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0))
            img_padded = cv2.resize(img_padded, (32, 32))
            img_padded = img_padded.astype("float32") / 255.0
            img_padded = np.expand_dims(img_padded, axis=-1)
            chars.append((img_padded, (x, y, w, h)))

    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    return boxes, chars

    # img = img.resize((32, 32))
    # img_array = np.asarray(img)
    # img_array = np.expand_dims(img_array, axis=0)
    # img_array = np.expand_dims(img_array, axis=-1)  
    # img_array = img_array / 255.0 
    # return img_array

def predict_letter(image_path):
    boxes, chars = preprocess_image(image_path)
    prediction = model.predict(chars)
    
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labels = [l for l in letters ]
    
    output = ""
    
    for (pred, (x, y, w, h)) in zip(prediction, boxes):
        i = np.argmax(pred)
        prob = pred[i]
        label = labels[i]
        print(label, ': ', prob)
        if prob > 0.85:
            output += label

    # predicted_class = np.argmax(prediction)
    # predicted_letter = chr(ord('A') + predicted_class)
    print(output)
    return output

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

if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
