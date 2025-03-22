import cv2

# Initialize the known distance from the camera to the object
KNOWN_DISTANCE = 24.0

# Initialize the known width of the object
KNOWN_WIDTH = 14.3

# Initialize the focal length of the camera
FOCAL_LENGTH = 0

def distance_to_camera(perWidth, focalLength):
    # Compute and return the distance from the camera to the object
    return (KNOWN_WIDTH * focalLength) / perWidth

# Initialize the OpenCV face detector
face_cascade = cv2.CascadeClassifier('D:/distance calculation/haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Calculate the focal length of the camera
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    FOCAL_LENGTH = (w * KNOWN_DISTANCE) / KNOWN_WIDTH
    break

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through all detected faces
    for (x, y, w, h) in faces:
        # Compute the width of the face in pixels
        perWidth = w

        # Compute and display the distance to the face
        inches = distance_to_camera(perWidth, FOCAL_LENGTH)
        cv2.putText(frame, "Distance: {:.2f} inches".format(inches), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()