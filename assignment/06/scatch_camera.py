import cv2
import numpy as np

# Get image using local webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    # Read Frame
    _, frame = cap.read()

    # Resize
    frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]), fx=0.5, fy=0.5)

    # If you press key `esc`, then this process will be terminate
    if cv2.waitKey(1) == 27:
        cv2.imwrite("scatch_result.png", img_paint)
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur
    gray_img = cv2.GaussianBlur(gray_img, (9, 9), 0)
    # Edge detect
    edges = cv2.Laplacian(gray_img, -1, None, 5)
    # Edge thresholding
    _, sketch = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY_INV)

    # Morphology: erode
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    sketch = cv2.erode(sketch, kernel)
    sketch = cv2.medianBlur(sketch, 5)
    rgb_sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    # Blur to image
    img_paint = cv2.blur(frame, (10, 10))
    img_paint = cv2.bitwise_and(img_paint, rgb_sketch)

    cv2.imshow("Image", img_paint)

cap.release()
cv2.destroyAllWindows()
