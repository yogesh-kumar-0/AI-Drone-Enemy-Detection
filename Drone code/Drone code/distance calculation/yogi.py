import cv2
import threading
import gmplot
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import face_recognition

# Constants
KNOWN_DISTANCE = 24.0  # inches
KNOWN_WIDTH = 14.3  # inches

# Initialize the face detector
face_cascade = cv2.CascadeClassifier('D:/distance calculation/haarcascade_frontalface_default.xml')
# Google Maps API key
API_KEY = 'AIzaSyDf7nz85ffWGOFJlAi4vkSYTZ9Pb1gNrJM'

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

# Function to update Google Map
def update_google_map(positions):
    # Default location: San Francisco
    gmap = gmplot.GoogleMapPlotter(37.7749, -122.4194, 13, apikey=API_KEY)

    if positions:
        latitudes, longitudes = zip(*[(pos[0], pos[1]) for pos in positions])
        gmap.scatter(latitudes, longitudes, '#FF0000', size=50, marker=False)
        gmap.draw("map1.html")

# Function to update 3D plot
def update_3d_plot(fig, ax, positions):
    ax.clear()

    if positions:
        latitudes, longitudes, distances = zip(*[(pos[0], pos[1], pos[2]) for pos in positions])
        ax.scatter(latitudes, longitudes, distances, c='r', marker='o')

    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Distance (inches)')
    plt.draw()

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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        with lock:
            people_positions = []
            for (x, y, w, h), face_encoding in zip(faces, face_encodings):
                matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
                if True in matches:
                    continue  # Skip known faces

                distance = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, w)
                cv2.putText(frame, f"Distance: {distance:.2f} inches", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Example: Mock positions around the center (for demonstration)
                latitude = 37.7749 + (x / frame.shape[1] * 0.01) - 0.005
                longitude = -122.4194 + (y / frame.shape[0] * 0.01) - 0.005
                people_positions.append((latitude, longitude, distance))

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to periodically update the map and 3D plot
def map_and_plot_update_thread(fig, ax):
    while True:
        with lock:
            update_google_map(people_positions)
            update_3d_plot(fig, ax, people_positions)
        time.sleep(2)  # Update every 2 seconds

# Shared data and lock for synchronization
people_positions = []
lock = threading.Lock()

# Start video capture thread
video_thread = threading.Thread(target=video_capture_thread)
video_thread.start()

# Initialize matplotlib plot
plt.ion()  # Enable interactive mode for Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Start map and plot update thread
map_thread = threading.Thread(target=map_and_plot_update_thread, args=(fig, ax))
map_thread.start()

video_thread.join()
map_thread.join()