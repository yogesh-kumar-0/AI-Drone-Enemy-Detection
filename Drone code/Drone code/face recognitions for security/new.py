import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
import face_recognition

# Constants
KNOWN_DISTANCE = 24.0  # Known distance from the camera to the object (in inches)
KNOWN_WIDTH = 14.3  # Known width of the object (in inches)
FRAME_RESIZE = 0.5  # Factor by which to resize the frame

# Initialize the face detector
face_cascade = cv2.CascadeClassifier('D:/distance calculation/haarcascade_frontalface_default.xml')

# Predefined known faces
known_faces_encodings = []
known_faces_names = []

# Load known face images and encode them
image1 = face_recognition.load_image_file("C:/Users/bindh/Pictures/photo/WhatsApp Image 2024-05-31 at 09.14.39_13c985cc.jpg")
encoding1 = face_recognition.face_encodings(image1)[0]
known_faces_encodings.append(encoding1)
known_faces_names.append("Person 1")

# image2 = face_recognition.load_image_file('path_to_image2.jpg')
# encoding2 = face_recognition.face_encodings(image2)[0]
# known_faces_encodings.append(encoding2)
# known_faces_names.append("Person 2")

# Function to calculate distance to camera
def distance_to_camera(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

# Function to capture video and detect faces
def video_capture_thread():
    global people_positions, FOCAL_LENGTH
    cap = cv2.VideoCapture(0)

    # Calculate the focal length using a known distance and object width
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    FOCAL_LENGTH = 0
    for (x, y, w, h) in faces:
        FOCAL_LENGTH = (w * KNOWN_DISTANCE) / KNOWN_WIDTH
        break

    frame_count = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if frame_count % 5 == 0:  # Perform face recognition every 5 frames
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            recognized_faces = [False] * len(faces)

            for i, (x, y, w, h) in enumerate(faces):
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
                    if True in matches:
                        recognized_faces[i] = True

        with lock:
            people_positions = []
            for i, (x, y, w, h) in enumerate(faces):
                if i < len(recognized_faces) and recognized_faces[i]:
                    continue  # Skip known faces

                distance = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, w / FRAME_RESIZE)
                cv2.putText(frame, f"Distance: {distance:.2f} inches", (int(x / FRAME_RESIZE), int(y / FRAME_RESIZE) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.rectangle(frame, (int(x / FRAME_RESIZE), int(y / FRAME_RESIZE)), (int((x + w) / FRAME_RESIZE), int((y + h) / FRAME_RESIZE)), (255, 0, 0), 2)

                # Assuming the person's position is at the center of the detected face
                person_position = ((x + w / 2) / FRAME_RESIZE, (y + h / 2) / FRAME_RESIZE, distance)
                people_positions.append(person_position)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# Function to update 3D plot
def update_plot():
    global people_positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    current_position = (0, 0, 0)

    while True:
        with lock:
            ax.clear()
            ax.scatter(current_position[0], current_position[1], current_position[2], color='blue', label='Current Position')

            for position in people_positions:
                ax.scatter(position[0], position[1], position[2], color='red')
                ax.plot([current_position[0], position[0]], [current_position[1], position[1]], [current_position[2], position[2]], 'k-')

            ax.set_xlabel('X Position (inches)')
            ax.set_ylabel('Y Position (inches)')
            ax.set_zlabel('Z Position (inches)')
            ax.set_title('3D Positions and Connections')
            ax.legend()
        
        plt.pause(1)  # Pause to allow the plot to update

# Shared data and lock for synchronization
people_positions = []
lock = threading.Lock()

# Start video capture and plotting threads
video_thread = threading.Thread(target=video_capture_thread)
plot_thread = threading.Thread(target=update_plot)

video_thread.start()
plot_thread.start()

video_thread.join()
plot_thread.join()