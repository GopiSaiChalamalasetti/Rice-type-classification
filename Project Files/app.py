from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import tensorflow_hub as hub
from werkzeug.utils import secure_filename

# Custom HubLayer used in the rice.h5 model
class HubLayer(tf.keras.layers.Layer):
    def __init__(self, hub_url, **kwargs):
        super().__init__(**kwargs)
        self.hub_layer = hub.KerasLayer(hub_url, trainable=False)

    def call(self, inputs):
        return self.hub_layer(inputs)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the rice classification model
model = load_model("rice.h5", custom_objects={'HubLayer': HubLayer}, compile=False)

# Rice class labels
labels = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag']

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Details route
@app.route('/details')
def details():
    return render_template('details.html')

# About Us page
@app.route('/aboutus.html')
def about():
    return render_template('aboutus.html')

# Contact Us page
@app.route('/contactus.html')
def contact():
    return render_template('contactus.html')

# Prediction result route

@app.route('/result', methods=['POST'])
def result():
    try:
        if 'image' not in request.files:
            return "No file part in the request", 400

        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = cv2.imread(filepath)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img)
        predicted_class = labels[np.argmax(prediction)]

        # Pass the image and result to template
        image_path = f"uploads/{filename}"

        return render_template('results.html',
                               prediction_text=f"Predicted Rice Type: {predicted_class}",
                               image_path=image_path)

    except Exception as e:
        return f"Error: {str(e)}"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
