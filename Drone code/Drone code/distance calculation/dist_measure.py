import numpy as np
import cv2

# Define object specific variables  
dist = 0
focal = 450
width = 4

# Find the distance from the camera
def get_dist(rectangle_params, image):
    # Find number of pixels covered
    pixels = rectangle_params[1][0]
    print(pixels)
    # Calculate distance
    dist = (width * focal) / pixels
    
    # Write on the image
    image = cv2.putText(image, 'Distance from Camera in CM:', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    image = cv2.putText(image, str(round(dist, 2)), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return image

# Extract Frames 
cap = cv2.VideoCapture(0)

# Basic constants for OpenCV functions
kernel = np.ones((3, 3), 'uint8')

cv2.namedWindow('Object Dist Measure', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Dist Measure', 700, 600)

# Loop to capture video frames
while True:
    ret, img = cap.read()
    if not ret:
        break

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Predefined mask for green color detection
    lower = np.array([37, 51, 24])
    upper = np.array([83, 104, 131])
    mask = cv2.inRange(hsv_img, lower, upper)

    # Remove extra garbage from image
    d_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    # Find the contours
    contours, hierarchy = cv2.findContours(d_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    for cnt in contours:
        # Check for contour area
        if 100 < cv2.contourArea(cnt) < 306000:
            # Draw a rectangle on the contour
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], -1, (255, 0, 0), 3)
            
            img = get_dist(rect, img)

    cv2.imshow('Object Dist Measure', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
