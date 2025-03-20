from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import cv2
import os

app = Flask(__name__)

@app.route('/compare', methods=['POST'])
def compare_faces():
    try:
        # Receive images from the request
        reference_image = request.files['reference_image']
        # Save images temporarily
        reference_path = './reference.jpg'
        
        reference_image.save(reference_path)
        
        rgb_frame = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

        # Perform face detection and anti-spoofing check
        face_objs = DeepFace.extract_faces(
            img_path=rgb_frame,
            enforce_detection=False,  # Do not enforce face detection
            detector_backend='opencv',  # Use OpenCV for face detection
            anti_spoofing=True  # Enable anti-spoofing
        )

        # Initialize the result
        is_real = True  # Assume the face is real by default

        # Check each detected face
        for face_obj in face_objs:
            if not face_obj.get("is_real", False):  # If the face is not real
                print("Detected spoofing attempt, skipping this face.")
                is_real = False
                break  

        # Return the result
        return jsonify({
            'antispoofingproperty': is_real,  
        })
    except Exception as e:
        # Log the error and return a 500 response
        print(f"Error during anti-spoofing check: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
