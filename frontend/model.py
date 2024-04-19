import logging
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ExifTags
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__, static_folder='frontend')
CORS(app)  # Ensure CORS is enabled

# Load the model
model_path = '/Users/vishnu/Desktop/viji files/sem 6/IR/Project/model_optimal.h5'
model = load_model(model_path)
weights_path = '/Users/vishnu/Desktop/viji files/sem 6/IR/Project/model_weights.weights.h5'
model.load_weights(weights_path)

client_id = '818b87c2df5a45f48d3705e8b930edd1'
client_secret = '25273da78dc24e98833288ba169f9691'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


playlist_ids = {
    'Positive': 'https://open.spotify.com/playlist/37i9dQZF1DX9XIFQuFvzM4?si=0be74a602a4f4614',
    'Neutral': 'https://open.spotify.com/playlist/37i9dQZF1EIWfUH5fvLgHm?si=4d27668ae34241ac',
    'Negative': 'https://open.spotify.com/playlist/37i9dQZF1DWSqBruwoIXkA?si=4d19eedbeb33434b',
    'disgust': 'https://open.spotify.com/playlist/37i9dQZF1EIUDtahSUEMYS?si=6c06a6505c8e4d62',
    'neutral': 'https://open.spotify.com/playlist/37i9dQZF1EIVVAaLvNJxGC?si=89e460a3337a47cb',

}

# Emotion to category mapping
emotion_to_category = {
    'happy': 'Positive',
    'surprised': 'Positive',
    'neutral': 'neutral',
    'anger': 'Neutral',
    'fear': 'Neutral',
    'sad': 'Negative',
    'disgust': 'disgust'
}

# Valence score ranges and sort orders for each tag
valence_rules = {
    'happy': (0.66, 1, False),
    'surprised': (0.5, 0.7, True),
    'sad': (0, 0.27, True),
    'disgust': (0.2, 0.4, True),
    'neutral': (0, 0.4, False),
    'anger': (0.4, 0.7, False),
    'fear': (0.2, 0.4, True)
}

from flask import Flask, send_from_directory
import os

app = Flask(__name__, static_url_path='', static_folder='frontend')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# The rest of your Flask app code...


@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    logging.info("Received a request!")
    if 'imageFile' not in request.files:
        logging.error("No image file provided.")
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['imageFile']
    if file:
        try:
            img = Image.open(file.stream)
            img = correct_image_orientation(img)
            img = process_image(img)
            emotion = predict_emotion_from_image(img)
            return jsonify(emotion=emotion)
        except Exception as e:
            logging.error(f"Error processing the image: {e}")
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file'}), 400

def correct_image_orientation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())
        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        logging.warning("No EXIF orientation data found.")
    return img

def process_image(img):
    img = img.convert('L')
    img = img.resize((48, 48))
    return np.array(img) / 255.0

def predict_emotion_from_image(img_array):
    img_array = img_array.reshape(1, 48, 48, 1)  # Add batch dimension
    result = model.predict(img_array)
    predicted_class_index = np.argmax(result, axis=1)
    label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
    return label_dict[predicted_class_index[0]]



if __name__ == '__main__':
    app.run(debug=True)
