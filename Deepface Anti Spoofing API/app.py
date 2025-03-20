from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import os
import numpy as np

app = Flask(__name__)

@app.route('/compare', methods=['POST'])
def compare_faces():
    try:
        # Receive images from the request
        reference_image = request.files['reference_image']
        
         # Convert FileStorage to NumPy array
        # file_bytes = np.frombuffer(reference_image.read(), np.uint8)
        # img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Perform face detection and anti-spoofing check
        face_objs = DeepFace.extract_faces(
            img_path=reference_image,  # Pass the image as a NumPy array
            enforce_detection=False,
            detector_backend='opencv',
            anti_spoofing=True
        )

        # Initialize the result
        is_real = True  # Assume the face is real by default

        # Check each detected face
        for face_obj in face_objs:
            if not face_obj.get("is_real", False): 
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
