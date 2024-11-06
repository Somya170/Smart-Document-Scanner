import cv2
import numpy as np

# Load the image
image = cv2.imread('ae.jpg')
if image is None:
    print("Image not found.")
    exit()

# Resize the image for faster processing (optional)
image = cv2.resize(image, (500, int(500 * image.shape[0] / image.shape[1])))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding to highlight the document edges
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variable to store the document contour
document_contour = None
max_area = 0

# Loop over contours to find the largest 4-sided contour
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:  # Filter small contours
        # Approximate the contour to reduce the number of points
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Check if the contour has four sides and is the largest found
        if len(approx) == 4 and area > max_area:
            document_contour = approx
            max_area = area

# If a document contour was found, draw it on the image
result = image.copy()
if document_contour is not None:
    cv2.drawContours(result, [document_contour], -1, (0, 255, 0), 2)
    print("Document boundary detected.")
else:
    print("No document boundary found.")

# Show the result
cv2.imshow("Document Boundary", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
