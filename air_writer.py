import cv2
import numpy as np
import time
import verification as ver

# Define the lower and upper boundaries for a color to be considered "blue"
blueLower = np.array([100, 150, 0], dtype="uint8")
blueUpper = np.array([140, 255, 255], dtype="uint8")

# Start video capture
cap = cv2.VideoCapture(0)

# Create a black image for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prev_center = None
mode = 'write'  # 'write' or 'erase'
pen_color = (255, 255, 255)  # White for writing
pen_thickness = 4  # Thinner pen
min_move_dist = 5  # Minimum distance to draw a line
eraser_radius = 30  # Size of the eraser

print("Press 's' to save, 'w' to write, 'e' to erase, and 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror the frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for blue color
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(contours) > 0:
        # Find the largest contour in the mask
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if mode == 'write':
                # Draw on the canvas if moved enough
                if prev_center is not None:
                    dist = np.linalg.norm(np.array(center) - np.array(prev_center))
                    if dist > min_move_dist:
                        cv2.line(canvas, prev_center, center, pen_color, pen_thickness)
                        prev_center = center
                else:
                    prev_center = center
            elif mode == 'erase':
                # Draw a big black circle at the current position
                cv2.circle(canvas, center, eraser_radius, (0, 0, 0), -1)
                prev_center = center
        else:
            prev_center = None
    else:
        prev_center = None

    # Overlay the canvas on the webcam feed for real-time feedback
    overlay = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.putText(overlay, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    if mode == 'erase' and center is not None:
        # Show the eraser circle on the overlay for feedback
        cv2.circle(overlay, center, eraser_radius, (0, 0, 0), 2)
    cv2.imshow("Air Writing (Overlay)", overlay)
    cv2.imshow("Canvas", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("drawing.png", canvas)
        print("Drawing saved as drawing.png")
        break
    elif key == ord('w'):
        mode = 'write'
        print("Switched to write mode.")
    elif key == ord('e'):
        mode = 'erase'
        print("Switched to erase mode.")
    elif key == ord('q'):
        print("Exiting without saving.")
        break

    time.sleep(0.01)  # Small delay to slow down the speed

cap.release()
cv2.destroyAllWindows()
verifacted_text = ver.prompt_LLM("drawing.png")
file = open('output.txt', 'w', encoding='utf-8')
file.write(verifacted_text)

