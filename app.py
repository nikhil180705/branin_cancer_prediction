from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
model = load_model("D:\\project\\brain tumor\\templates\\brain_tumor_model.h5")  # Use the path to your saved model

# Define the categories
categories = ["glioma", "meningioma", "notumor", "pituitary"]

# Create the 'uploads' directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Ensure the image size matches the input size of the model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = request.files['file']
        img_path = os.path.join(UPLOAD_FOLDER, img.filename)
        img.save(img_path)
        
        # Prepare the image
        img_array = prepare_image(img_path)
        
        # Predict using the loaded model
        predictions = model.predict(img_array)
        os.remove(img_path)  # Remove the image after prediction
        
        predicted_class = categories[np.argmax(predictions[0])]
        
        
        result = {"prediction": predicted_class}
        return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
