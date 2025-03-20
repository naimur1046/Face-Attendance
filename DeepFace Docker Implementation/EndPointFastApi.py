from fastapi import FastAPI, File, UploadFile
import requests
import os
import cv2

app = FastAPI()

@app.post("/compare-video")
async def compare_video(reference_image: UploadFile, video_file: UploadFile):
    try:
        # Save reference image
        reference_path = './reference.jpg'
        with open(reference_path, 'wb') as ref:
            ref.write(reference_image.file.read())

        # Save video
        video_path = './input_video.mp4'
        with open(video_path, 'wb') as vid:
            vid.write(video_file.file.read())

        # Extract frames from video
        frame_folder = './frames'
        video_to_frames(video_path, frame_folder)

        # Compare each frame with the reference image
        results = []
        for frame_file in os.listdir(frame_folder):
            frame_path = os.path.join(frame_folder, frame_file)
            with open(frame_path, 'rb') as frame:
                files = {
                    'reference_image': open(reference_path, 'rb'),
                    'comparison_image': frame
                }
                response = requests.post("http://localhost:5000/compare", files=files)
                results.append(response.json())

        return {"results": results}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup temporary files
        if os.path.exists(reference_path):
            os.remove(reference_path)
        if os.path.exists(video_path):
            os.remove(video_path)

# Function for frame extraction
def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    video_capture.release()
