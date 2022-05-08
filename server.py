from flask import Flask, render_template,request
app = Flask(__name__)

import os
UPLOAD_FOLDER = './static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np

loaded_model = load_model('model.h5')

classes = {
    0 : 'Covid',
    1 : 'Viral Pneumonia',
    2 : 'Normal'
}

def predictClass(imgPath):
  img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
  img = cv2.resize(img, (128,128))
  img = np.array(img, dtype='float32')
  img = img/255
  img = np.reshape(img, (1,128,128,3))
  label = loaded_model.predict(img)
  predict_labels = np.argmax(label, axis=1)
  return classes[predict_labels[0]]


@app.route('/', methods=['GET', 'POST'])
def home():
  if request.method == 'POST':
    file1 = request.files['file1']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
    file1.save(path)

    p = predictClass(path)
    return render_template('home.html', prediction=p, path=path)
  return render_template('home.html')