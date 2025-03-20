import cv2
import os
import numpy as np
import face_recognition

# Path to the folder containing reference images
image_folder_path = "./reference_images"

# Ensure the reference images folder exists
if not os.path.exists(image_folder_path):
    print(f"Error: Reference images folder '{image_folder_path}' does not exist.")
    exit(1)

# Load and preprocess reference images
known_face_encodings = []
known_face_names = []

print("Loading and encoding reference images...")
for image_file in os.listdir(image_folder_path):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder_path, image_file)
        try:
            # Load the image
            image = face_recognition.load_image_file(image_path)

            # Get face encodings for the image
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_face_encodings.append(encodings[0])  # Use the first face encoding
                known_face_names.append(os.path.splitext(image_file)[0])  # File name without extension
            else:
                print(f"No face found in {image_file}. Skipping.")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

print(f"Loaded {len(known_face_encodings)} reference images.")

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

    # Convert the frame to RGB as face_recognition expects RGB format
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the frame's encoding with known encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        # Find the best match
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            print(f"Detected: {name} (Distance: {face_distances[best_match_index]:.2f})")

            # Display the name on the frame
            top, right, bottom, left = face_location
            cv2.rectangle(resized_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(resized_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            print("No match found.")

    # Display the frame
    cv2.imshow('Live Video Stream', resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
