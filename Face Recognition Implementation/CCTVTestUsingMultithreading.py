import cv2
import os
import numpy as np
import face_recognition
from multiprocessing import Process, Queue, cpu_count
from threading import Thread

def frame_dropper(frame_queue, processed_frame_queue):
    while True:
        if frame_queue.qsize() > 2:  # Drop frames if the queue grows too large
            frame_queue.get()
        frame = frame_queue.get()
        processed_frame_queue.put(frame)

def process_frames(processed_frame_queue, display_queue, known_face_encodings, known_face_names):
    while True:
        frame = processed_frame_queue.get()
        if frame is None:
            break

        resized_frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                print(f"Detected: {name} (Distance: {face_distances[best_match_index]:.2f})")

                top, right, bottom, left = face_location
                cv2.rectangle(resized_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(resized_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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

    known_face_encodings = []
    known_face_names = []

    print("Loading and encoding reference images...")
    for image_file in os.listdir(image_folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder_path, image_file)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(os.path.splitext(image_file)[0])
                else:
                    print(f"No face found in {image_file}. Skipping.")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

    print(f"Loaded {len(known_face_encodings)} reference images.")

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
        p = Process(target=process_frames, args=(processed_frame_queue, display_queue, known_face_encodings, known_face_names))
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
