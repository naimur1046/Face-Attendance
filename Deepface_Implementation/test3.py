import cv2
import os
import numpy as np
from deepface import DeepFace
import threading

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
            embedding = DeepFace.represent(img_path=image_path, model_name='ArcFace')[0]['embedding']
            known_face_embeddings.append(np.array(embedding))
            known_face_names.append(os.path.splitext(image_file)[0])  # File name without extension
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

print(f"Loaded {len(known_face_embeddings)} reference images.")

# Start video capture
video_capture = cv2.VideoCapture("rtsp://rndteam:rnddev1234%@192.168.6.76/media/video1")

if not video_capture.isOpened():
    print("Error: Unable to access the camera.")
    exit(1)

print("Starting live video stream. Press 'q' to quit.")

# Frame Processing Pipeline
# Multi-threaded function to process frames
def process_frame(frame):
    try:
        # Resize frame for faster processing
        # resized_frame = cv2.resize(frame, (640, 480))

        # Convert the frame to RGB as DeepFace expects RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract faces from the frame
        face_objs = DeepFace.extract_faces(
            img_path=rgb_frame,
            enforce_detection=False,  # Prevent errors if no face is detected
            detector_backend='opencv',  # Specify backend (default is fine)
            anti_spoofing=True  # Enable anti-spoofing
        )

        for face_obj in face_objs:
            if not face_obj.get("is_real", True):
                print("Detected spoofing attempt, skipping this face.")
                continue

            # Get embedding for the frame
            frame_embedding = DeepFace.represent(img_path=rgb_frame, model_name='ArcFace')[0]['embedding']
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
                cv2.putText(frame, f"{name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("No match found.")

    except Exception as e:
        print(f"Error analyzing frame: {e}")

    # Display the frame
    cv2.imshow('Live Video Stream', frame)

# Threaded frame reading and processing
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Unable to capture video frame.")
        break

    # Start a new thread for processing each frame
    threading.Thread(target=process_frame, args=(frame,)).start()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
