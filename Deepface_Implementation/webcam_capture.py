import cv2
import requests
import os

# Define the URL of the Flask API
api_url = 'http://localhost:5000/match-images'

# Open webcam
cap = cv2.VideoCapture(0)

# Capture frames and send to API
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Show the captured frame (optional)
    cv2.imshow('Webcam', frame)

    # Save the current frame to a temporary file
    frame_filename = 'temp_frame.jpg'
    cv2.imwrite(frame_filename, frame)
    
    # Open the reference image (you may want to provide your own reference image)
    reference_image = 'reference_image.jpg'

    # Prepare data for the API request
    files = {
        'reference_image': open(reference_image, 'rb'),
        'matching_image': open(frame_filename, 'rb')
    }

    # Send POST request to the API
    response = requests.post(api_url, files=files)

    # Print the response from the API
    print(response.json())

    # Stop the webcam capture on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
