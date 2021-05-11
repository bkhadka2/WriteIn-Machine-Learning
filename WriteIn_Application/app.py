from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
import cv2
import base64
import tensorflow as tf
import os
print('Tensorflow imported')
print('Imported Necessary modules')


def model():
    return keras.models.load_model('Saved_Model')


def predict(img):
    modelr = model()
    predictions = modelr.predict(img)
    predictionCharacter = np.argmax(predictions)
    predictionDict = dict([(0, 'A'), (1, 'B'), (2, 'C'), (3, 'D'), (4, 'E'), (5, 'F'), (6, 'G'), (7, 'H'), (8, 'I'), (9, 'J'), (10, 'K'), (11, 'L'), (12, 'M'), (13, 'N'), (14, 'O'), (15, 'P'), (16, 'Q'), (17, 'R'), (18, 'S'), (19, 'T'), (20, 'U'), (21, 'V'), (22, 'W'), (23, 'X'), (24, 'Y'), (25, 'Z')])
    return predictionDict[predictionCharacter]


app = Flask(__name__)

@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/Predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        data = request.get_json()
        imagebase64 = data['image']
        imgbytes = base64.b64decode(imagebase64)
        with open("writeInImg.jpg","wb") as decodedImage:
            decodedImage.write(imgbytes)
        img = cv2.imread('./writeInImg.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        newNormalizedImg = tf.keras.utils.normalize(resized, axis=1)
        newNormalizedImg = np.array(newNormalizedImg).reshape(-1, 28, 28, 1)

        return jsonify({
            'prediction': predict(newNormalizedImg),
            'status': True
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)