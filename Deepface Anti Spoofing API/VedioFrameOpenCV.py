import cv2
import os

def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load video
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Save frame as image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    video_capture.release()
    print(f"Saved {frame_count} frames to {output_folder}")

# Example usage
video_path = "sample_video.mp4"  # Path to your video
output_folder = "./frames"
video_to_frames(video_path, output_folder)
