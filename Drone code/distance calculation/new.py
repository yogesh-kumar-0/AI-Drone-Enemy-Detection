import numpy as np
import cv2

# Define object specific variables
focal_length = 450  # Example focal length in pixels
real_object_width = 4  # Real width of the object in centimeters

# Function to calculate distance from camera
def calculate_distance(rectangle_params, image):
    # Number of pixels covered by the object
    pixels = rectangle_params[1][0]
    print(f"Width in pixels: {pixels}")

    # Calculate distance using the formula
    distance = (real_object_width * focal_length) / pixels

    # Write distance on the image
    image = cv2.putText(image, 'Distance from Camera in CM:', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    image = cv2.putText(image, f"{round(distance, 2)} cm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return image

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define constants for OpenCV functions
kernel = np.ones((3, 3), 'uint8')

cv2.namedWindow('Object Distance Measurement', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Distance Measurement', 700, 600)

# Loop to capture video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define mask for green color detection
    lower_green = np.array([37, 51, 24])
    upper_green = np.array([83, 104, 131])
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Remove extra noise from the image
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    # Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    for contour in contours:
        # Check for significant contour area
        if 100 < cv2.contourArea(contour) < 306000:
            # Draw a rectangle around the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], -1, (255, 0, 0), 3)

            # Calculate distance and annotate the frame
            frame = calculate_distance(rect, frame)

    # Display the result
    cv2.imshow('Object Distance Measurement', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
