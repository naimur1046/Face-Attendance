import cv2
import os
import numpy as np
from deepface import DeepFace
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils

# Path to the folder containing reference images
image_folder_path = "./reference_images"

# Ensure the reference images folder exists
if not os.path.exists(image_folder_path):
    print(f"Error: Reference images folder '{image_folder_path}' does not exist.")
    exit(1)

# Load and preprocess reference images
known_face_embeddings = []
known_face_names = []

print("Loading and encoding reference images...")
for image_file in os.listdir(image_folder_path):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder_path, image_file)
        try:
            # Get embedding for each image
            embedding = DeepFace.represent(img_path=image_path, model_name='Facenet512')[0]['embedding']
            known_face_embeddings.append(np.array(embedding))
            known_face_names.append(os.path.splitext(image_file)[0])  # File name without extension
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

print(f"Loaded {len(known_face_embeddings)} reference images.")

# Initialize Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    """
    Calculate Eye Aspect Ratio (EAR) to detect blinks.
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_liveness(frame, gray):
    """
    Perform liveness detection using blink detection.
    """
    faces = detector(gray)
    for rect in faces:
        # Get facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract left and right eyes
        left_eye = shape[36:42]
        right_eye = shape[42:48]

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Draw eyes on the frame for visualization
        cv2.polylines(frame, [left_eye], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.polylines(frame, [right_eye], isClosed=True, color=(0, 255, 0), thickness=1)

        # Blink detection threshold
        if avg_ear < 0.25:  # Blink detected
            return True
    return False

# Start video capture
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Unable to access the camera.")
    exit(1)

print("Starting live video stream. Press 'q' to quit.")

while True:
    # Capture a single frame from the live stream
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Unable to capture video frame.")
        break

    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (640, 480))

    # Convert the frame to grayscale and RGB
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Check for liveness (blink detection)
    is_live = detect_liveness(resized_frame, gray)

    if is_live:
        try:
            # Get embedding for the frame
            frame_embedding = DeepFace.represent(img_path=rgb_frame, model_name='Facenet512')[0]['embedding']
            frame_embedding = np.array(frame_embedding)

            # Compare the frame's embedding with known embeddings
            best_match = None
            min_distance = float('inf')
            threshold = 0.4  # Adjust this threshold as needed

            for idx, known_embedding in enumerate(known_face_embeddings):
                # Compute cosine similarity
                cosine_distance = np.dot(frame_embedding, known_embedding) / (
                    np.linalg.norm(frame_embedding) * np.linalg.norm(known_embedding)
                )
                distance = 1 - cosine_distance  # Convert similarity to distance
                if distance < min_distance:
                    min_distance = distance
                    best_match = idx

            # Check if the best match is within the acceptable threshold
            if min_distance < threshold and best_match is not None:
                name = known_face_names[best_match]
                print(f"Detected: {name} (Distance: {min_distance:.2f})")

                # Display the name on the frame
                cv2.putText(resized_frame, f"{name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("No match found.")
        except Exception as e:
            print(f"Error analyzing frame: {e}")
    else:
        cv2.putText(resized_frame, "Liveness Check Failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Live Video Stream', resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
