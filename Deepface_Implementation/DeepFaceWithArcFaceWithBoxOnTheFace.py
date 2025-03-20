import cv2
import os
import numpy as np
from deepface import DeepFace

# Choose the model to use (e.g., 'VGG-Face', 'Facenet', 'Facenet512', 'ArcFace', 'SFace', etc.)
model_name = "ArcFace"  

# Path to the folder containing reference images
image_folder_path = "./reference_images"

# Ensure the reference images folder exists
if not os.path.exists(image_folder_path):
    print(f"Error: Reference images folder '{image_folder_path}' does not exist.")
    exit(1)

# Load and preprocess reference images
known_face_embeddings = []
known_face_names = []

print(f"Loading and encoding reference images using model: {model_name}...")
for image_file in os.listdir(image_folder_path):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder_path, image_file)
        try:
            # Get embedding for each image
            embedding = DeepFace.represent(img_path=image_path, model_name=model_name,anti_spoofing=True)[0]['embedding']
            known_face_embeddings.append(np.array(embedding))
            known_face_names.append(os.path.splitext(image_file)[0])  # File name without extension
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

print(f"Loaded {len(known_face_embeddings)} reference images.")

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

    # Convert the frame to RGB as DeepFace expects RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Detect faces in the frame
        detections = DeepFace.analyze(img_path=rgb_frame, actions=['age'], enforce_detection=False)

        # Loop through detected faces
        for detection in detections:
            # Extract bounding box coordinates
            x, y, w, h = detection['region']['x'], detection['region']['y'], detection['region']['w'], detection['region']['h']

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract face embedding
            frame_embedding = DeepFace.represent(img_path=rgb_frame[y:y + h, x:x + w], model_name=model_name)[0]['embedding']
            frame_embedding = np.array(frame_embedding)

            # Compare the frame's embedding with known embeddings
            best_match = None
            min_distance = float('inf')
            threshold = 0.5  # Adjust this threshold as needed

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
                cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                print("No match found.")
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    except Exception as e:
        print(f"Error analyzing frame: {e}")

    # Display the frame
    cv2.imshow('Live Video Stream', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
