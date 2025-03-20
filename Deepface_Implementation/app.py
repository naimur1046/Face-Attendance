from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Initialize MediaPipe Face Detection model
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Upload folder
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to detect faces in an image using MediaPipe
def detect_face(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(image_rgb)
        if results.detections:
            return True
    return False

# API route for image matching
@app.route('/match-images', methods=['POST'])
def match_images():
    if 'reference_image' not in request.files or 'matching_image' not in request.files:
        return jsonify({"error": "Both reference_image and matching_image are required."}), 400

    reference_image = request.files['reference_image']
    matching_image = request.files['matching_image']
    
    # Save uploaded images to disk
    ref_filename = secure_filename(reference_image.filename)
    match_filename = secure_filename(matching_image.filename)

    reference_image_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_filename)
    matching_image_path = os.path.join(app.config['UPLOAD_FOLDER'], match_filename)

    reference_image.save(reference_image_path)
    matching_image.save(matching_image_path)

    # Detect faces in both images
    if detect_face(reference_image_path) and detect_face(matching_image_path):
        return jsonify({"message": "Images match!"}), 200
    else:
        return jsonify({"message": "Images do not match."}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
