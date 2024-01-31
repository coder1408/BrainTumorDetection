from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'Project\BrainTumorDetection\static\uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg'}

# Load the trained model
model_path = r'Project\BrainTumorDetection\brain-tumor1.h5'  # Replace with the actual path to your trained model
model = load_model(model_path)

# Preprocess image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index-tensorflow.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(file_path)

        # Make predictions
        input_tensor = preprocess_image(file_path)

        # Use the loaded model for prediction
        with tf.device('/cpu:0'):  # Use CPU to avoid GPU memory issues
            prediction = model.predict(input_tensor)

        class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        predicted_class = class_labels[np.argmax(prediction)]

        return render_template('index-tensorflow.html', prediction=predicted_class, image_path=file_path)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
