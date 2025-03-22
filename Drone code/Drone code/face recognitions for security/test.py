import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
import face_recognition
from queue import Queue
from matplotlib.animation import FuncAnimation

# Constants
KNOWN_DISTANCE = 24.0  # Known distance from the camera to the object (in inches)
KNOWN_WIDTH = 14.3  # Known width of the object (in inches)

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

# Function to calculate distance to camera
def distance_to_camera(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

# Function to capture video and detect faces
def video_capture_thread(people_queue):
    global FOCAL_LENGTH
    cap = cv2.VideoCapture(0)
    
    time.sleep(2)  # Warm-up time for the camera

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
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        people_positions = []
        for (x, y, w, h), face_encoding in zip(faces, face_encodings):
            matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
            if True in matches:
                continue  # Skip known faces

            distance = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, w)
            cv2.putText(frame, f"Distance: {distance:.2f} inches", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Assuming the person's position is at the center of the detected face
            person_position = (x + w / 2, y + h / 2, distance)
            people_positions.append(person_position)

        # Send positions to the queue
        if people_queue.full():
            people_queue.get()
        people_queue.put(people_positions)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to update 3D plot in the main thread
def update_plot(frame, people_queue, ax, current_position):
    if not people_queue.empty():
        people_positions = people_queue.get()

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

# Shared data queue for synchronization
people_queue = Queue(maxsize=1)

# Start video capture thread
video_thread = threading.Thread(target=video_capture_thread, args=(people_queue,))
video_thread.start()

# Start 3D plot update in the main thread
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
current_position = (0, 0, 0)

ani = FuncAnimation(fig, update_plot, fargs=(people_queue, ax, current_position), interval=1000)
plt.show()

video_thread.join()
