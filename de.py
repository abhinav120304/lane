import cv2
import numpy as np

def detect_lanes(image):
    """Detects lanes in an image and returns the processed frame."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 30, 120)  # Adjust threshold values if needed

    # Define Region of Interest (ROI)
    height, width = image.shape[:2]
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[ 
        (0, height), (width // 2, height // 2), (width, height) 
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Line Transform for lane detection
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, minLineLength=40, maxLineGap=100)

    if lines is None:
        print("⚠️ No lanes detected! Adjusting parameters...")
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 20, minLineLength=20, maxLineGap=150)
        if lines is None:
            print("❌ Still no lanes detected.")
            return image

    # Draw detected lane lines
    output = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Green lane lines

    return output

def process_image(image_path):
    """Processes a static image for lane detection."""
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Image not found!")
        return
    
    lane_frame = detect_lanes(image)
    cv2.imshow("Lane Detection - Image", lane_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_webcam():
    """Processes live webcam feed for lane detection."""
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return
    
    print("🎥 Webcam lane detection started... Press 'Q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break

        lane_frame = detect_lanes(frame)

        cv2.imshow("Lane Detection - Live Webcam", lane_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = input("Choose mode - (1) Image or (2) Live Video: ")
    
    if mode == "1":
        image_path = input("Enter image path (e.g., test_images/image.jpg): ")
        process_image(image_path)
    elif mode == "2":
        process_webcam()
    else:
        print("❌ Invalid option! Choose 1 for image or 2 for live video.")
