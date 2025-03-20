import cv2
import pandas as pd
import os
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# Load embeddings
def load_embeddings(filename="embeddings.pkl"):
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    else:
        raise FileNotFoundError(f"Embeddings file '{filename}' not found. Please run the face registration script first.")

# Recognize face and mark attendance
def recognize_and_mark_attendance(embeddings, app):
    cap = cv2.VideoCapture(1)  # Open webcam
    attendance_file = "attendance.csv"

    # Load existing attendance or initialize a new DataFrame
    if os.path.exists(attendance_file):
        attendance = pd.read_csv(attendance_file)
    else:
        attendance = pd.DataFrame(columns=["Name", "Time"])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break

        # Detect faces in the frame
        faces = app.get(frame)
        for face in faces:
            input_embedding = face.embedding.reshape(1, -1)
            best_match = None
            best_similarity = 0

            # Compare input embedding with saved embeddings
            for name, saved_embeddings in embeddings.items():
                for saved_embedding in saved_embeddings:
                    similarity = cosine_similarity(input_embedding, np.array(saved_embedding).reshape(1, -1))[0][0]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name

            if best_match and best_similarity > 0.5:  # Threshold for recognition
                print(f"Recognized: {best_match}, Similarity: {best_similarity:.2f}")

                # Mark attendance if not already marked for the day
                if not (attendance["Name"] == best_match).any():
                    attendance = pd.concat(
                        [attendance, pd.DataFrame([{"Name": best_match, "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])],
                        ignore_index=True,
                    )
                    attendance.to_csv(attendance_file, index=False)

                # Draw bounding box and label
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{best_match} ({best_similarity:.2f})", (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Attendance System", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Attendance system closed.")

if __name__ == "__main__":
    try:
        embeddings = load_embeddings()  # Load saved embeddings
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))
        recognize_and_mark_attendance(embeddings, app)
    except Exception as e:
        print(f"Error: {e}")
