from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import os

app = Flask(__name__)

@app.route('/compare', methods=['POST'])
def compare_faces():
    try:
        # Receive images from the request
        reference_image = request.files['reference_image']
        comparison_image = request.files['comparison_image']
        
        # Save images temporarily
        reference_path = './reference.jpg'
        comparison_path = './comparison.jpg'
        reference_image.save(reference_path)
        comparison_image.save(comparison_path)

        # Get embeddings for both images
        reference_representation = DeepFace.represent(img_path=reference_path, model_name='VGG-Face')
        comparison_representation = DeepFace.represent(img_path=comparison_path, model_name='VGG-Face')

        if not reference_representation or not comparison_representation:
            raise ValueError("Could not generate embeddings for the images.")

        reference_embedding = np.array(reference_representation[0]['embedding'])
        comparison_embedding = np.array(comparison_representation[0]['embedding'])

        # Calculate cosine similarity
        cosine_similarity = np.dot(reference_embedding, comparison_embedding) / (
            np.linalg.norm(reference_embedding) * np.linalg.norm(comparison_embedding)
        )

        # Threshold for matching (higher cosine similarity means more similar)
        threshold = 0.4  
        match = cosine_similarity > (1 - threshold)

        # Return result
        return jsonify({
            'match': bool(match),
            'cosine_similarity': float(cosine_similarity)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files
        if os.path.exists(reference_path):
            os.remove(reference_path)
        if os.path.exists(comparison_path):
            os.remove(comparison_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
