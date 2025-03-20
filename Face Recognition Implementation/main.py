import cv2
import os
import dlib
import numpy as np

def load_dlib_models():
    """Load dlib models for face detection, landmark prediction, and face recognition."""
    try:
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        return face_detector, shape_predictor, face_rec_model
    except Exception as e:
        print(f"Error loading dlib models: {e}")
        exit(1)

def load_reference_images(folder_path, face_detector, shape_predictor, face_rec_model):
    """Load and encode reference images from a folder."""
    if not os.path.exists(folder_path):
        print(f"Error: Reference images folder '{folder_path}' does not exist.")
        exit(1)

    known_embeddings = []
    known_names = []
    print("Loading and encoding reference images...")

    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_file)
            try:
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = face_detector(rgb_image)

                if len(faces) > 0:
                    shape = shape_predictor(rgb_image, faces[0])
                    face_chip = dlib.get_face_chip(rgb_image, shape)
                    embedding = np.array(face_rec_model.compute_face_descriptor(face_chip))

                    known_embeddings.append(embedding)
                    known_names.append(os.path.splitext(image_file)[0])
                else:
                    print(f"No face detected in {image_file}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

    print(f"Loaded {len(known_embeddings)} reference images.")
    return known_embeddings, known_names

def recognize_faces(frame, face_detector, shape_predictor, face_rec_model, known_embeddings, known_names):
    """Detect and recognize faces in a given frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb_frame)

    for face in faces:
        try:
            shape = shape_predictor(rgb_frame, face)
            face_chip = dlib.get_face_chip(rgb_frame, shape)
            frame_embedding = np.array(face_rec_model.compute_face_descriptor(face_chip))

            best_match = None
            min_distance = float('inf')
            threshold = 0.6

            for idx, known_embedding in enumerate(known_embeddings):
                distance = np.linalg.norm(frame_embedding - known_embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_match = idx

            if min_distance < threshold and best_match is not None:
                name = known_names[best_match]
                cv2.putText(frame, f"{name}", (face.left(), face.top() - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (face.left(), face.top() - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error analyzing face: {e}")

    return frame

def main():
    # Load dlib models
    face_detector, shape_predictor, face_rec_model = load_dlib_models()

    # Path to the folder containing reference images
    image_folder_path = "./reference_images"

    # Load and encode reference images
    known_embeddings, known_names = load_reference_images(image_folder_path, face_detector, shape_predictor, face_rec_model)

    # Start video capture
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Unable to access the camera.")
        exit(1)

    print("Starting live video stream. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture video frame.")
            break

        resized_frame = cv2.resize(frame, (640, 480))
        processed_frame = recognize_faces(resized_frame, face_detector, shape_predictor, face_rec_model, known_embeddings, known_names)

        cv2.imshow('Live Video Stream', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
