import cv2
import os
import numpy as np
from deepface import DeepFace
from multiprocessing import Process, Queue, cpu_count
from threading import Thread

def frame_dropper(frame_queue, processed_frame_queue):
    while True:
        if frame_queue.qsize() > 4:  # Adjust frame-dropping threshold
            frame_queue.get()
        frame = frame_queue.get()
        processed_frame_queue.put(frame)

def process_frames(processed_frame_queue, display_queue, known_face_embeddings, known_face_names):
    while True:
        frame = processed_frame_queue.get()
        if frame is None:
            break

        resized_frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        try:
            # Detect and analyze faces
            detected_faces = DeepFace.extract_faces(img_path=rgb_frame, detector_backend='opencv')

            for face_data in detected_faces:
                face_embedding = DeepFace.represent(img_path=face_data["face_image"], model_name="ArcFace")

                # Compare the detected face with known embeddings
                distances = [np.linalg.norm(np.array(face_embedding) - np.array(known_embedding)) for known_embedding in known_face_embeddings]
                min_distance = min(distances)

                if min_distance < 0.6:  # Set a threshold for matching
                    best_match_index = distances.index(min_distance)
                    name = known_face_names[best_match_index]

                    print(f"Detected: {name} (Distance: {min_distance:.2f})")

                    # Draw bounding box and label
                    box = face_data['facial_area']
                    top, left, bottom, right = box[1], box[0], box[3], box[2]
                    cv2.rectangle(resized_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(resized_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error processing frame: {e}")

        display_queue.put(resized_frame)

def display_frames(display_queue):
    while True:
        frame = display_queue.get()
        if frame is None:
            break

        cv2.imshow('CCTV Video Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    # Path to the folder containing reference images
    image_folder_path = "./reference_images"

    if not os.path.exists(image_folder_path):
        print(f"Error: Reference images folder '{image_folder_path}' does not exist.")
        return

    known_face_embeddings = []
    known_face_names = []

    print("Loading and encoding reference images...")
    for image_file in os.listdir(image_folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder_path, image_file)
            try:
                embedding = DeepFace.represent(img_path=image_path, model_name="Facenet")
                known_face_embeddings.append(embedding)
                known_face_names.append(os.path.splitext(image_file)[0])
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

    print(f"Loaded {len(known_face_embeddings)} reference images.")

    cctv_rtsp_url = "rtsp://rndteam:rnddev1234%@192.168.6.76/media/video1"
    video_capture = cv2.VideoCapture(cctv_rtsp_url)

    if not video_capture.isOpened():
        print("Error: Unable to access the RTSP stream.")
        return

    print("Starting CCTV video stream. Press 'q' to quit.")

    frame_queue = Queue(maxsize=10)
    processed_frame_queue = Queue(maxsize=10)
    display_queue = Queue(maxsize=10)

    frame_dropper_thread = Thread(target=frame_dropper, args=(frame_queue, processed_frame_queue))
    frame_dropper_thread.daemon = True
    frame_dropper_thread.start()

    process_pool = []
    for _ in range(cpu_count() - 1):
        p = Process(target=process_frames, args=(processed_frame_queue, display_queue, known_face_embeddings, known_face_names))
        p.daemon = True
        p.start()
        process_pool.append(p)

    display_thread = Thread(target=display_frames, args=(display_queue,))
    display_thread.daemon = True
    display_thread.start()

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Unable to capture video frame.")
                break

            if not frame_queue.full():
                frame_queue.put(frame)
    except KeyboardInterrupt:
        pass

    for _ in process_pool:
        processed_frame_queue.put(None)

    for p in process_pool:
        p.join()

    display_queue.put(None)
    display_thread.join()
    video_capture.release()

if __name__ == "__main__":
    main()
