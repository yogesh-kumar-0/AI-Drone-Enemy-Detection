import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import time

# Constants
KNOWN_DISTANCE = 24.0  # Known distance from the camera to the object (in inches)
KNOWN_WIDTH = 14.3  # Known width of the object (in inches)

# Initialize the face detector
face_cascade = cv2.CascadeClassifier('D:/distance calculation/haarcascade_frontalface_default.xml')

# Function to calculate distance to camera
def distance_to_camera(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

# Function to capture video and detect faces
def video_capture_thread():
    global people_positions, FOCAL_LENGTH
    cap = cv2.VideoCapture(0)

    # Calculate the focal length using a known distance and object width
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    FOCAL_LENGTH = 0
    for (x, y, w, h) in faces:
        FOCAL_LENGTH = (w * KNOWN_DISTANCE) / KNOWN_WIDTH
        break

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        with lock:
            people_positions = []
            for (x, y, w, h) in faces:
                distance = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, w)
                cv2.putText(frame, f"Distance: {distance:.2f} inches", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Assuming the person's position is at the center of the detected face
                person_position = (x + w / 2, y + h / 2, distance)
                people_positions.append(person_position)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
        
        plt.pause(0.1)  # Pause to allow the plot to update

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