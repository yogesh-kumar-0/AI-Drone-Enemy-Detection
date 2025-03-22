import cv2
import threading
import gmplot
import time

# Constants
KNOWN_DISTANCE = 24.0  # Known distance from the camera to the object (in inches)
KNOWN_WIDTH = 14.3  # Known width of the object (in inches)

# Initialize the face detector
face_cascade = cv2.CascadeClassifier('D:/distance calculation/haarcascade_frontalface_default.xml')

# Google Maps API key
API_KEY = 'AIzaSyDf7nz85ffWGOFJlAi4vkSYTZ9Pb1gNrJM'

# Function to calculate distance to camera
def distance_to_camera(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

# Function to update Google Map
def update_google_map(positions):
    # Default location: San Francisco
    gmap = gmplot.GoogleMapPlotter(37.7749, -122.4194, 13, apikey=API_KEY)

    if positions:
        latitudes, longitudes = zip(*positions)
        gmap.scatter(latitudes, longitudes, '#FF0000', size=50, marker=False)
        gmap.plot(latitudes, longitudes, 'cornflowerblue', edge_width=2.5)

    gmap.draw("map1.html")

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
                # This example assumes all positions are at the default location (for demonstration purposes)
                person_position = (37.7749, -122.4194)  # Default to San Francisco
                people_positions.append(person_position)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to periodically update the map
def map_update_thread():
    while True:
        with lock:
            update_google_map(people_positions)
        time.sleep(1)  # Update the map every 5 seconds

# Shared data and lock for synchronization
people_positions = []
lock = threading.Lock()

# Start video capture and map updating threads
video_thread = threading.Thread(target=video_capture_thread)
map_thread = threading.Thread(target=map_update_thread)

video_thread.start()
map_thread.start()

video_thread.join()
map_thread.join()